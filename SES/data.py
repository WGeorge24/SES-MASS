"""Dataset helpers for DREAMS / EDF20 / EDF78 / SHHS."""

from __future__ import annotations

from . import _env  # noqa: F401

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from utils import load_dataloader, load_dataset, load_train_dataset  # noqa: E402

from .config import DATASETS, NUM_EPOCHS_WINDOW, STEP, DatasetConfig


def list_mat_files(cfg: DatasetConfig) -> List[Path]:
    data_dir = cfg.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    files = sorted(p for p in data_dir.glob(cfg.file_glob) if p.is_file())
    if not files:
        hint = ""
        if cfg.name == "shhs":
            hint = " Run: python myTest/preprocess.py --dataset shhs"
        raise FileNotFoundError(f"No files matching {cfg.file_glob} in {data_dir}.{hint}")
    return files


def subject_id(path: Path, split: str) -> str:
    """One recording group key. Sleep-EDF uses subject digits; SHHS is 1 file = 1 subject."""
    name = path.name
    if name.startswith("SC4") and len(name) >= 5:
        return name[3:5]
    if split == "loso":
        return path.stem.replace("subject", "")
    return path.stem


def group_by_subject(files: Sequence[Path], split: str) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        groups[subject_id(p, split)].append(p)
    return dict(groups)


def create_folds(subject_ids: Sequence[str], num_folds: int = 20, seed: int = 2025) -> List[List[str]]:
    subjects = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    fold_size = max(1, len(subjects) // num_folds)
    folds = [subjects[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    remainder = subjects[num_folds * fold_size:]
    for i, sid in enumerate(remainder):
        folds[i % num_folds].append(sid)
    folds = [f for f in folds if f]
    return folds


def iter_splits(
    cfg: DatasetConfig, seed: int = 2025, fold: Optional[int] = None, max_folds: Optional[int] = None
):
    files = list_mat_files(cfg)
    groups = group_by_subject(files, cfg.split)
    if cfg.split == "loso":
        subject_keys = sorted(groups.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
        folds = [[k] for k in subject_keys]
    else:
        folds = create_folds(list(groups.keys()), num_folds=20, seed=seed)

    indices = list(range(len(folds)))
    if fold is not None:
        indices = [fold]
    if max_folds is not None:
        indices = indices[:max_folds]

    for fold_idx in indices:
        test_subjects = folds[fold_idx]
        train_subjects = [s for i, f in enumerate(folds) if i != fold_idx for s in f]
        train_files = [str(p) for s in train_subjects for p in groups[s]]
        test_files = [str(p) for s in test_subjects for p in groups[s]]
        yield fold_idx, train_files, test_files, test_subjects


def _stack_train(files: Sequence[str], data_key: str, length: int = NUM_EPOCHS_WINDOW, step: int = STEP):
    arrays = [load_train_dataset(f, length, data_key, "label", step) for f in files]
    data = np.concatenate([a[0] for a in arrays], axis=0)
    label = np.concatenate([a[1] for a in arrays], axis=0)
    trans = np.concatenate([a[2] for a in arrays], axis=0)
    return data, label, trans


def _stack_test(files: Sequence[str], data_key: str, length: int = NUM_EPOCHS_WINDOW):
    arrays = [load_dataset(f, length, data_key, "label") for f in files]
    data = np.concatenate([a[0] for a in arrays], axis=0)
    label = np.concatenate([a[1] for a in arrays], axis=0)
    trans = np.concatenate([a[2] for a in arrays], axis=0)
    return data, label, trans


def load_fold_tensors(
    train_files: Sequence[str],
    test_files: Sequence[str],
    data_key: str,
    batch_size: int,
    seed: int = 2025,
    device: str = "cpu",
):
    data_train, label_train, trans_train = _stack_train(train_files, data_key)
    data_test, label_test, trans_test = _stack_test(test_files, data_key)

    mean = float(np.mean(data_train))
    std = float(np.std(data_train) + 1e-8)
    data_train = (data_train - mean) / std
    data_test = (data_test - mean) / std

    arr = trans_train.reshape(-1)
    freq_0 = float(np.mean(arr == 0))
    freq_1 = float(np.mean(arr == 1))
    trans_weights = torch.tensor([freq_0, freq_1], dtype=torch.float32)

    data_train_t = torch.tensor(data_train)
    data_test_t = torch.tensor(data_test)
    label_train_t = torch.tensor(label_train)
    label_test_t = torch.tensor(label_test)
    trans_train_t = torch.tensor(trans_train)
    trans_test_t = torch.tensor(trans_test)

    train_iter, test_iter = load_dataloader(
        data_train_t, data_test_t, label_train_t, label_test_t,
        trans_train_t, trans_test_t, batch_size=batch_size, seed=seed,
    )
    stats = {
        "mean": mean,
        "std": std,
        "train_shape": tuple(data_train.shape),
        "test_shape": tuple(data_test.shape),
        "trans_weights": trans_weights,
    }
    return train_iter, test_iter, stats


def assert_dataset_ready(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name}. Choose from {list(DATASETS)}")
    cfg = DATASETS[name]
    if not cfg.enabled:
        raise RuntimeError(f"{name} is not enabled ({cfg.notes})")
    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Missing {cfg.data_dir}")
    files = list_mat_files(cfg)
    if name == "shhs" and len(files) < 2:
        raise RuntimeError(
            f"SHHS needs at least 2 preprocessed recordings in {cfg.data_dir} "
            f"(found {len(files)}). Run: python myTest/preprocess.py --dataset shhs"
        )
    return cfg
