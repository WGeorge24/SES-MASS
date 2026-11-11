"""CLI for SES: equal-interval whole-epoch retention.

From the MASS-main repo root:

    python -m SES --schedule
    python -m SES --smoke
    python -m SES --dataset dreams --budget 0.72
    python -m SES --dataset dreams --budget 0.40
    python -m SES --dataset dreams --budget 0.10
    python -m SES --dataset dreams --budget 1.0
    python -m SES --all
    python -m SES --ablation
    python -m SES --dataset dreams --pi random --budget 0.40
    python -m SES --dataset dreams --pi head --budget 0.40
    python -m SES --dataset dreams --pi uniform --budget 0.40
    python -m SES --dataset dreams --budget 0.40 --fast --fold 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from . import _env  # noqa: F401
from .config import ABLATION_POLICIES, BUDGET_GRID, DATASETS, PI_CHOICES, RunConfig, canonical_budget
from . import data as D
from . import sampling as S
from . import train as T
from .metrics import format_metrics, mean_std
from .model import SES
from .paths import PACKAGE_DIR, ROOT

import torch


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ses")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SES: equal-interval 30s epoch retention on the equal-interval 1 s grid"
    )
    parser.add_argument("--dataset", default="dreams", choices=list(DATASETS.keys()))
    parser.add_argument(
        "--pi", default="equal", choices=list(PI_CHOICES),
        help="equal=SES-MASS (default); "
             "random=SES-Random; head=SES-Head; uniform=SES-Uniform; "
             "random_inter=equal-interval seconds with random whole epochs",
    )
    parser.add_argument(
        "--budget", type=float, default=0.40,
        help="Nominal signal integrity: 1.0, 0.72, 0.40, or 0.10",
    )
    parser.add_argument(
        "--mix_random", type=float, default=0.25,
        help="Train-only fraction of batches whose 1s patches are drawn at random. "
             "Used by --pi equal and --pi random_inter. "
             "random, head, and uniform ignore this and keep mix at 0.",
    )
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fast", action="store_true", help="1 fold, 2 epochs — sanity check, not a paper result")
    parser.add_argument("--smoke", action="store_true", help="Check the timetable and a dummy forward/backward")
    parser.add_argument("--schedule", action="store_true", help="Print the phase-0 acquisition timetable and exit")
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 16 paper jobs in order: dreams, edf20, edf78, shhs × 0.72, 0.40, 0.10, 1.0",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Table-3 ablation: dreams, edf20, edf78, shhs × random, head, uniform, equal at nominal 40%",
    )
    return parser.parse_args()


def make_run_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(
        dataset=args.dataset,
        pi=args.pi,
        budget=canonical_budget(args.budget),
        mix_random=args.mix_random,
        fold=args.fold,
        max_folds=args.max_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        fast=args.fast,
    )
    if args.epochs is not None:
        cfg.train.total_epochs = args.epochs
        cfg.train.warmup_epochs = max(1, args.epochs // 10)
    if args.fast:
        cfg.train.total_epochs = min(2, cfg.train.total_epochs)
        cfg.train.warmup_epochs = 0
        cfg.fold = 0 if cfg.fold is None else cfg.fold
        cfg.max_folds = 1
    cfg.train.seed = args.seed
    if cfg.pi in ("random", "head", "uniform"):
        cfg.mix_random = 0.0
    return cfg


def build_model(cfg: RunConfig, mean, std, device: torch.device) -> SES:
    policy = S.build_policy(
        cfg.pi,
        cfg.budget,
        num_epochs=cfg.model.num_epoches,
        num_patches=cfg.model.num_patches,
        seed=cfg.seed,
        mix_random=cfg.mix_random,
    )
    model = SES(
        hparams=cfg.model,
        policy=policy,
        mean=mean,
        std=std,
        seed=cfg.seed,
    )
    return model.to(device)


def print_schedules() -> None:
    S.assert_deployment_schedules()
    for budget in BUDGET_GRID:
        print(S.format_schedule(S.deployment_schedule(budget)))
        print()


def _assert_unique_patch_rows(patch_vis: torch.Tensor, keep_patches: int) -> None:
    flat = patch_vis.reshape(-1, patch_vis.shape[-1])
    for row in flat:
        if int(torch.unique(row).numel()) != keep_patches:
            raise AssertionError(f"random intra row is not a unique {keep_patches}-subset: {row.tolist()}")


def smoke_test(cfg: RunConfig, logger: logging.Logger) -> None:
    S.assert_deployment_schedules()
    device = T.get_device(cfg.device)
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    model = build_model(cfg, mean, std, device)
    b, t, p, f = 4, cfg.model.num_epoches, cfg.model.num_patches, cfg.model.feat_dim
    eeg = torch.randn(b, t, p, f, device=device)
    label = torch.randint(0, 5, (b, t), device=device)
    trans = (torch.rand(b, t, device=device) > 0.8).long()
    model.train()
    out = model(eeg, transition=trans, return_aux=True)
    if out["stage"].shape != (b * t, 5):
        raise AssertionError(out["stage"].shape)
    loss = T.mass_loss(
        out["stage"], label.reshape(-1), out["transition"], out["transition_label"],
        torch.tensor([0.8, 0.2], device=device), cfg.train,
    )
    loss.backward()

    model.eval()
    with torch.no_grad():
        first = model(eeg, transition=trans, return_aux=True)
        second = model(eeg, transition=trans, return_aux=True)
    pol_a = first["policy"]
    pol_b = second["policy"]
    same_epochs = torch.equal(pol_a.epoch_visible, pol_b.epoch_visible)
    same_patches = torch.equal(pol_a.patch_visible, pol_b.patch_visible)
    schedule = S.deployment_schedule(cfg.budget)
    got_epochs = pol_a.epoch_visible[0].detach().cpu().tolist()
    got_patches = pol_a.patch_visible[0, 0].detach().cpu().tolist()
    if cfg.pi == "equal":
        if not same_epochs or not same_patches:
            raise AssertionError("Deployment timetable changed between two eval forwards.")
        if got_epochs != schedule["epoch_keep"] or got_patches != schedule["patch_keep"]:
            raise AssertionError(
                f"Eval indices differ from the published timetable.\n"
                f"epochs {got_epochs}\npatches {got_patches}"
            )
        if len({tuple(row.tolist()) for row in pol_a.epoch_visible}) != 1:
            raise AssertionError("A batch does not share one epoch timetable.")
        if abs(pol_a.overall_ratio - schedule["realized_ratio"]) > 1e-8:
            raise AssertionError(pol_a.overall_ratio)
    elif cfg.pi == "random_inter":
        # 100% keeps every epoch, so the two draws are identical by construction.
        if schedule["keep_epochs"] < schedule["num_epochs"] and same_epochs:
            raise AssertionError("random_inter eval redrew the same epoch subset twice.")
    elif cfg.pi in ("random", "head", "uniform"):
        if not same_epochs or got_epochs != schedule["epoch_keep"]:
            raise AssertionError(
                f"{cfg.pi} moved the phase-0 epoch timetable: {got_epochs}"
            )
        if abs(pol_a.overall_ratio - schedule["realized_ratio"]) > 1e-8:
            raise AssertionError(pol_a.overall_ratio)
        if cfg.pi == "head":
            head = list(range(schedule["keep_patches"]))
            if got_patches != head or not same_patches:
                raise AssertionError(f"head patches changed: {got_patches}")
        elif cfg.pi == "uniform":
            if got_patches != schedule["patch_keep"] or not same_patches:
                raise AssertionError(f"uniform patches changed: {got_patches}")
        else:
            _assert_unique_patch_rows(pol_a.patch_visible, schedule["keep_patches"])
            if schedule["keep_patches"] < schedule["num_patches"] and same_patches:
                raise AssertionError("random intra redrew the same 1s subset twice.")
    else:
        raise AssertionError(f"smoke has no checks for {cfg.pi}")

    # Training may shift phase, but the kept counts stay exact.
    model.train()
    with torch.no_grad():
        train_out = model.policy.sample(b, device)
    if train_out.epoch_visible.shape[1] != schedule["keep_epochs"]:
        raise AssertionError(train_out.epoch_visible.shape)
    if train_out.patch_visible.shape[-1] != schedule["keep_patches"]:
        raise AssertionError(train_out.patch_visible.shape)
    if cfg.pi == "equal" and schedule["epoch_phase_span_train"] == 1:
        train_epochs = train_out.epoch_visible[0].detach().cpu().tolist()
        if train_epochs != schedule["epoch_keep"]:
            raise AssertionError(f"Single-phase epoch grid drifted in training: {train_epochs}")
    if cfg.pi == "equal" and schedule["epoch_phase_span_train"] > 1:
        seen = set()
        for _ in range(40):
            sample = model.policy.sample(1, device)
            seen.add(tuple(int(v) for v in sample.epoch_visible[0].detach().cpu().tolist()))
        if tuple(schedule["epoch_keep"]) not in seen or len(seen) < 2:
            raise AssertionError("Training epoch phases did not cover the deploy grid.")
    if cfg.pi in ("random", "head", "uniform"):
        train_epochs = train_out.epoch_visible[0].detach().cpu().tolist()
        train_patches = train_out.patch_visible[0, 0].detach().cpu().tolist()
        if train_epochs != schedule["epoch_keep"]:
            raise AssertionError(f"{cfg.pi} training moved the phase-0 epoch table: {train_epochs}")
        if cfg.pi == "head" and train_patches != list(range(schedule["keep_patches"])):
            raise AssertionError(f"head training patches changed: {train_patches}")
        if cfg.pi == "uniform" and train_patches != schedule["patch_keep"]:
            raise AssertionError(f"uniform training shifted phase: {train_patches}")
        if cfg.pi == "random":
            _assert_unique_patch_rows(train_out.patch_visible, schedule["keep_patches"])
        if cfg.mix_random != 0.0:
            raise AssertionError(f"{cfg.pi} must run with mix_random 0, got {cfg.mix_random}")

    shown_epochs = "redrawn" if cfg.pi == "random_inter" else schedule["epoch_keep"]
    logger.info(
        "Smoke OK  pi=%s nominal=%.2f realized=%.6f keep=%sx%s loss=%.4f deploy_epochs=%s",
        cfg.pi, cfg.budget, schedule["realized_ratio"],
        schedule["keep_epochs"], schedule["keep_patches"], float(loss.detach()),
        shown_epochs,
    )


def run_cv(cfg: RunConfig, logger: logging.Logger, out_dir: Path) -> dict:
    ds = D.assert_dataset_ready(cfg.dataset)
    device = T.get_device(cfg.device)
    batch_size = cfg.batch_size or ds.batch_size
    fold_metrics: List[dict] = []
    n_mat = len(D.list_mat_files(ds))
    logger.info(
        "dataset=%s files=%s batch_size=%s data_key=%s split=%s",
        ds.name, n_mat, batch_size, ds.data_key, ds.split,
    )
    if cfg.dataset == "shhs" and n_mat < 329:
        logger.warning(
            "SHHS expected 329 shhs1-*.mat (AttnSleep subset), found %s. "
            "Finish: python myTest/preprocess.py --dataset shhs",
            n_mat,
        )

    for fold_idx, train_files, test_files, test_subjects in D.iter_splits(
        ds, seed=cfg.seed, fold=cfg.fold, max_folds=cfg.max_folds
    ):
        logger.info(
            "==== %s fold %s n_train=%s n_test=%s test_subjects=%s ====",
            cfg.dataset, fold_idx, len(train_files), len(test_files), test_subjects,
        )
        train_iter, test_iter, stats = D.load_fold_tensors(
            train_files, test_files, ds.data_key, batch_size, cfg.seed, str(device)
        )
        logger.info(
            "train %s test %s mean=%.4f std=%.4f",
            stats["train_shape"], stats["test_shape"], stats["mean"], stats["std"],
        )
        model = build_model(cfg, torch.tensor(stats["mean"]), torch.tensor(stats["std"]), device)
        metrics = T.train_one_fold(
            model, train_iter, test_iter, cfg, fold_idx,
            stats["trans_weights"], out_dir, logger,
        )
        metrics["test_subjects"] = list(map(str, test_subjects))
        fold_metrics.append(metrics)
        del model, train_iter, test_iter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = _summarize(fold_metrics)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": _cfg_dict(cfg), "folds": fold_metrics, "summary": summary}, f, indent=2)
    logger.info("==== %s summary ====\n%s", cfg.run_tag, json.dumps(summary, indent=2))
    return summary


def _summarize(fold_metrics: List[dict]) -> dict:
    keys = ["acc", "mf1", "kappa", "mgm", "obs_ratio", "obs_ratio_std"]
    out = {}
    for key in keys:
        vals = [m[key] for m in fold_metrics if key in m]
        mu, sd = mean_std(vals)
        out[key] = {"mean": mu, "std": sd}
    f1_stack = [m["f1_per_class"] for m in fold_metrics if "f1_per_class" in m]
    if f1_stack:
        import numpy as np
        arr = np.asarray(f1_stack, dtype=float)
        out["f1_per_class_mean"] = arr.mean(axis=0).tolist()
        out["stage_names"] = ["W", "N1", "N2", "N3", "REM"]
    return out


def _cfg_dict(cfg: RunConfig) -> dict:
    schedule = S.deployment_schedule(cfg.budget)
    return {
        "dataset": cfg.dataset,
        "pi": cfg.pi,
        "budget_nominal": cfg.budget,
        "mix_random": cfg.mix_random,
        "epochs": cfg.train.total_epochs,
        "seed": cfg.seed,
        "run_tag": cfg.run_tag,
        "schedule": schedule,
    }


ALL_DATASETS = ("dreams", "edf20", "edf78", "shhs")
ALL_BUDGETS = (0.72, 0.40, 0.10, 1.00)


def launch_run(cfg: RunConfig) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = PACKAGE_DIR / "results" / f"{cfg.run_tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = S.deployment_schedule(cfg.budget)
    schedule["pi"] = cfg.pi
    schedule["mix_random"] = cfg.mix_random
    if cfg.pi == "head":
        schedule["intra_patch_keep"] = list(range(schedule["keep_patches"]))
        schedule["intra_train_phase"] = False
    elif cfg.pi == "uniform":
        schedule["intra_patch_keep"] = list(schedule["patch_keep"])
        schedule["intra_train_phase"] = False
    elif cfg.pi == "random":
        schedule["intra_patch_keep"] = "random each forward"
        schedule["intra_train_phase"] = False
    elif cfg.pi == "equal":
        schedule["intra_train_phase"] = True
    with open(out_dir / "schedule.json", "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)
    logger = build_logger(out_dir / "train.log")
    logger.info("Repo root: %s", ROOT)
    logger.info("Run config: %s", json.dumps(_cfg_dict(cfg), indent=2))
    logger.info("Deployment timetable:\n%s", S.format_schedule(schedule))
    logger.info("Output: %s", out_dir)
    run_cv(cfg, logger, out_dir)
    return out_dir


def run_queue(args: argparse.Namespace, jobs: List[tuple], queue_prefix: str) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    queue_dir = PACKAGE_DIR / "results" / f"{queue_prefix}_{stamp}"
    queue_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(queue_dir / "queue.log")
    logger.info("Queue of %s jobs. A failed job does not stop the rest.", len(jobs))
    if args.fast:
        logger.info(" --fast is on: each job is 1 fold and 2 epochs, not a paper result.")
    finished = []
    for index, (dataset, budget, pi) in enumerate(jobs, start=1):
        logger.info(
            "==== %s/%s  dataset=%s  budget=%.2f  pi=%s ====",
            index, len(jobs), dataset, budget, pi,
        )
        args.dataset = dataset
        args.budget = budget
        args.pi = pi
        T.set_seed(args.seed)
        cfg = make_run_config(args)
        started = time.time()
        try:
            out_dir = launch_run(cfg)
            status = "ok"
            detail = str(out_dir)
        except Exception:
            status = "failed"
            detail = ""
            logger.exception("Job failed: %s B%.2f %s", dataset, budget, pi)
        elapsed = time.time() - started
        finished.append({
            "dataset": dataset,
            "budget": budget,
            "pi": pi,
            "status": status,
            "output": detail,
            "seconds": round(elapsed, 1),
        })
        logger.info("Job %s/%s %s  elapsed %.1f min", index, len(jobs), status, elapsed / 60.0)
        with open(queue_dir / "queue.json", "w", encoding="utf-8") as f:
            json.dump(finished, f, indent=2)
    failed = [item for item in finished if item["status"] != "ok"]
    logger.info("Queue done. ok=%s failed=%s  log=%s", len(finished) - len(failed), len(failed), queue_dir)
    if failed:
        raise SystemExit(1)


def run_all(args: argparse.Namespace) -> None:
    jobs = [(dataset, budget, "equal") for dataset in ALL_DATASETS for budget in ALL_BUDGETS]
    run_queue(args, jobs, "_queue")


def run_ablation(args: argparse.Namespace) -> None:
    jobs = [(dataset, 0.40, pi) for dataset in ALL_DATASETS for pi in ABLATION_POLICIES]
    run_queue(args, jobs, "_ablation")


def main():
    _configure_stdio()
    args = parse_args()
    if args.schedule:
        print_schedules()
        return
    if args.all and args.ablation:
        raise SystemExit("--all 和 --ablation 只能选一个。")
    if (args.all or args.ablation) and (args.smoke or args.schedule):
        raise SystemExit("队列不能和 --smoke 或 --schedule 一起用。")
    if args.all:
        run_all(args)
        return
    if args.ablation:
        run_ablation(args)
        return

    T.set_seed(args.seed)
    cfg = make_run_config(args)

    if args.smoke:
        logger = build_logger(PACKAGE_DIR / "results" / "_smoke.log")
        logger.info("Repo root: %s", ROOT)
        for budget in BUDGET_GRID:
            for pi in PI_CHOICES:
                cfg.pi = pi
                cfg.budget = budget
                cfg.mix_random = 0.0 if pi in ("random", "head", "uniform") else 0.25
                smoke_test(cfg, logger)
        logger.info("All smoke tests passed.")
        return

    launch_run(cfg)


if __name__ == "__main__":
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from SES.run import main as _main
        _main()
    else:
        main()
