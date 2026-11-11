"""Evaluation metrics (MASS protocol: ACC / MF1 / Kappa / MGm)."""

from __future__ import annotations

from . import _env  # noqa: F401

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import NUM_CLASSES, STAGE_NAMES


def gmean_from_cm(cm: np.ndarray, recall: np.ndarray) -> tuple:
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    spec = tn / np.clip(tn + fp, 1e-8, None)
    g_per = np.sqrt(np.clip(spec * recall, 0, None))
    return float(np.mean(g_per)), g_per


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = NUM_CLASSES) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    acc_per = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    mgm, g_per = gmean_from_cm(cm, rec_per)
    return {
        "acc": acc,
        "mf1": mf1,
        "kappa": kappa,
        "mgm": mgm,
        "acc_per_class": acc_per.tolist(),
        "recall_per_class": rec_per.tolist(),
        "f1_per_class": f1_per.tolist(),
        "gmean_per_class": g_per.tolist(),
        "cm": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def mean_std(values: List[float]) -> tuple:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def format_metrics(metrics: Dict) -> str:
    lines = [
        f"ACC={metrics['acc']:.4f}  MF1={metrics['mf1']:.4f}  "
        f"Kappa={metrics['kappa']:.4f}  MGm={metrics['mgm']:.4f}"
    ]
    for i, name in enumerate(STAGE_NAMES):
        lines.append(
            f"  {name}: P={metrics['acc_per_class'][i]:.4f}  "
            f"R={metrics['recall_per_class'][i]:.4f}  F1={metrics['f1_per_class'][i]:.4f}"
        )
    return "\n".join(lines)
