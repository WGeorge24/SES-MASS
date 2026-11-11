"""SES training / evaluation: MASS loss, select best checkpoint by ACC."""

from __future__ import annotations

from . import _env  # noqa: F401

import copy
import json
import logging
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import trange

from . import paths as _paths  # noqa: F401
from .config import NUM_CLASSES, RunConfig, TrainHParams
from .metrics import evaluate_predictions, format_metrics
from .model import SES


def set_seed(seed: int = 2025) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, hp: TrainHParams):
    try:
        from lion_pytorch import Lion
        return Lion(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    except Exception:
        return torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)


class CosineAnnealingWarmupRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, warmup_steps=0, max_lr=1e-3, min_lr=1e-6, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
            return [self.min_lr + (self.max_lr - self.min_lr) * scale for _ in self.optimizer.param_groups]
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.first_cycle_steps - self.warmup_steps)
        return [
            self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for _ in self.optimizer.param_groups
        ]


def mass_loss(pred, label, transition_pred, transition_label, trans_weights, hp: TrainHParams):
    """Exact MASS training loss: CE + 2 * (1 - cosine) + 0.5 * transition CE."""
    criterion = torch.nn.CrossEntropyLoss()
    loss1 = criterion(pred, label)
    one_hot = F.one_hot(label, num_classes=NUM_CLASSES).float()
    loss2 = 1.0 - F.cosine_similarity(pred, one_hot).mean()
    if transition_pred is not None and transition_pred.numel() > 0 and transition_label.numel() > 0:
        loss3 = F.cross_entropy(transition_pred, transition_label, weight=trans_weights)
    else:
        loss3 = pred.new_zeros(())
    return loss1 + hp.cosine_weight * loss2 + hp.transition_weight * loss3


@torch.no_grad()
def evaluate(model: SES, loader, device: torch.device) -> Dict:
    model.eval()
    y_true, y_pred, y_boundary = [], [], []
    costs = []
    for data, label, transition in loader:
        data = data.float().to(device)
        label = label.long().to(device)
        transition = transition.long().to(device)
        out = model(data, return_aux=True)
        preds = out["stage"].argmax(dim=-1)
        y_true.extend(label.reshape(-1).cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_boundary.extend(transition.reshape(-1).cpu().tolist())
        pol = out.get("policy")
        if pol is not None:
            costs.append(pol.overall_ratio)
    metrics = evaluate_predictions(y_true, y_pred)
    metrics["obs_ratio"] = float(np.mean(costs)) if costs else 1.0
    metrics["obs_ratio_std"] = float(np.std(costs)) if costs else 0.0
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_boundary"] = y_boundary
    return metrics


def train_one_fold(
    model: SES,
    train_iter,
    test_iter,
    cfg: RunConfig,
    fold: int,
    trans_weights: torch.Tensor,
    out_dir: Path,
    logger: logging.Logger,
) -> Dict:
    hp = cfg.train
    device = next(model.parameters()).device
    trans_weights = trans_weights.float().to(device)
    optimizer = build_optimizer(model, hp)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=hp.total_epochs, warmup_steps=hp.warmup_epochs,
        max_lr=hp.lr, min_lr=hp.min_lr,
    )

    best_acc = -1.0
    best_metrics = None
    best_state = None

    for ep in trange(hp.total_epochs, desc=f"fold {fold}"):
        model.train()
        total = 0.0
        n_batch = 0
        for data, label, transition in train_iter:
            data = data.float().to(device)
            label = label.long().to(device).view(-1)
            transition = transition.long().to(device)
            out = model(data, transition=transition, return_aux=True)
            loss = mass_loss(
                out["stage"], label, out["transition"], out["transition_label"], trans_weights, hp
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            n_batch += 1
        scheduler.step()
        metrics = evaluate(model, test_iter, device)
        logger.info(
            "Fold %s epoch %s  loss=%.4f  %s  obs=%.4f (std %.6f)",
            fold, ep + 1, total / max(n_batch, 1),
            format_metrics(metrics).split("\n")[0],
            metrics["obs_ratio"],
            metrics["obs_ratio_std"],
        )
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_metrics = {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred", "y_boundary")}
            best_metrics["epoch"] = ep + 1
            best_metrics["y_true"] = metrics["y_true"]
            best_metrics["y_pred"] = metrics["y_pred"]
            best_metrics["y_boundary"] = metrics["y_boundary"]
            best_state = copy.deepcopy(model.state_dict())

    if best_metrics is None:
        raise RuntimeError("No evaluation metrics were recorded. Check --epochs.")

    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt = out_dir / "checkpoints" / f"fold{fold}_acc{best_acc:.4f}.pth"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt)
        best_metrics["checkpoint"] = str(ckpt)

    pred_path = out_dir / "preds" / f"fold{fold}.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_path,
        y_true=np.asarray(best_metrics["y_true"]),
        y_pred=np.asarray(best_metrics["y_pred"]),
        y_boundary=np.asarray(best_metrics["y_boundary"]),
    )
    serializable = {k: v for k, v in best_metrics.items() if k not in ("y_true", "y_pred", "y_boundary")}
    with open(out_dir / f"fold{fold}.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Fold %s best:\n%s", fold, format_metrics(best_metrics))
    return serializable
