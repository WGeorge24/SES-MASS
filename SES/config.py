"""SES: equal-interval epoch retention on the equal-interval 1 s grid.

Signal budgets stay on the MASS Table-1 geometry. Only the placement
of the kept 30 s epochs changes: a fixed equal-interval timetable replaces
the per-night random draw. ``re = 0`` (keep every epoch) is the 100% setting
alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
NUM_CLASSES = 5
NUM_PATCHES = 30
NUM_EPOCHS_WINDOW = 32
FEAT_DIM = 128
STEP = 32

PI_CHOICES = ("equal", "random", "head", "uniform", "random_inter")
# Table-3 ablations. Epochs stay on the phase-0 equal-interval timetable.
# random/head/uniform do not shift phase and do not use mix_random.
ABLATION_POLICIES = ("random", "head", "uniform", "equal")
BUDGET_GRID: Tuple[float, ...] = (1.00, 0.72, 0.40, 0.10)

# Nominal signal integrity -> (intra mask ra, inter mask re).
# MASS MaskGenerator uses ceil, so the realized visible fraction is
# 72%: 24/30 patches x 28/32 epochs = 0.70
# 40%: 15/30 patches x 25/32 epochs = 0.390625
# 10%:  6/30 patches x 16/32 epochs = 0.10
# 100%: 30/30 x 32/32 = 1
PAPER_TABLE1_MASK: Dict[float, Tuple[float, float]] = {
    0.72: (0.2, 0.1),
    0.40: (0.5, 0.2),
    0.10: (0.8, 0.5),
}


def canonical_budget(budget: float) -> float:
    """Map a CLI value onto 1.00 / 0.72 / 0.40 / 0.10."""
    value = float(budget)
    for known in BUDGET_GRID:
        if abs(value - known) <= 1e-6:
            return known
    if abs(value - 0.70) <= 1e-6:
        raise ValueError(
            "0.70 is the realized ratio of the nominal 72% setting "
            "(24/30 patches x 28/32 epochs = 0.70). Use --budget 0.72. "
            "SES does not use the all-epochs geometry from newSAS."
        )
    raise ValueError(
        f"Unsupported budget {budget}. Choose one of 1.0, 0.72, 0.40, 0.10."
    )


def budget_to_mass_mask_ratios(budget: float) -> Tuple[float, float]:
    """(intra_mask_ratio, inter_mask_ratio) for a nominal Table-1 budget."""
    budget = canonical_budget(budget)
    if budget >= 0.999:
        return 0.0, 0.0
    return PAPER_TABLE1_MASK[budget]


def masked_count(mask_ratio: float, num_tokens: int) -> int:
    return int(math.ceil(num_tokens * float(mask_ratio) - 1e-9))


def visible_count(mask_ratio: float, num_tokens: int) -> int:
    return num_tokens - masked_count(mask_ratio, num_tokens)


def phase_span(keep: int, num_tokens: int) -> int:
    """Integer phases inside one step, matching the SES intra-epoch jitter.

    ``span = max(1, N // K)``. When the step is shorter than 2 tokens there
    is a single integer phase, so train and deployment share one timetable.
    """
    if keep >= num_tokens:
        return 1
    return max(1, num_tokens // int(keep))


@dataclass
class DatasetConfig:
    name: str
    rel_dir: str
    data_key: str
    split: str
    batch_size: int
    file_glob: str
    enabled: bool = True
    notes: str = ""

    @property
    def data_dir(self) -> Path:
        return ROOT / self.rel_dir


DATASETS: Dict[str, DatasetConfig] = {
    "dreams": DatasetConfig("dreams", "dataset/DREAMS", "PSD", "loso", 8, "subject*.mat"),
    "edf20": DatasetConfig("edf20", "dataset/EDF20", "PSD100", "subject20", 16, "SC4*.mat"),
    "edf78": DatasetConfig("edf78", "dataset/EDF78", "PSD100", "subject20", 64, "SC4*.mat"),
    "shhs": DatasetConfig(
        "shhs", "dataset/SHHS", "PSD", "subject20", 128, "shhs1-*.mat",
        enabled=True, notes="329 SHHS-1 recordings (AttnSleep subset), C4-A1, 20-fold.",
    ),
}


@dataclass
class ModelHParams:
    embed_dim: int = 128
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    num_patches: int = NUM_PATCHES
    num_epoches: int = NUM_EPOCHS_WINDOW
    global_encoder_depth: int = 4
    intra_encoder_depth: int = 4
    inter_encoder_depth: int = 2
    inter_hidden_dim: int = 256
    feat_dim: int = FEAT_DIM


@dataclass
class TrainHParams:
    lr: float = 1e-4
    weight_decay: float = 0.01
    total_epochs: int = 100
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    seed: int = 2025
    grad_clip: float = 5.0
    cosine_weight: float = 2.0
    transition_weight: float = 0.5
    select_metric: str = "acc"


@dataclass
class RunConfig:
    dataset: str = "dreams"
    pi: str = "equal"
    budget: float = 0.40
    mix_random: float = 0.25
    fold: Optional[int] = None
    max_folds: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    seed: int = 2025
    device: str = "cuda"
    fast: bool = False
    model: ModelHParams = field(default_factory=ModelHParams)
    train: TrainHParams = field(default_factory=TrainHParams)

    def __post_init__(self) -> None:
        self.budget = canonical_budget(self.budget)

    @property
    def run_tag(self) -> str:
        return f"{self.dataset}_{self.pi}_B{self.budget:.2f}"

    @property
    def keep_patches(self) -> int:
        intra, _ = budget_to_mass_mask_ratios(self.budget)
        return visible_count(intra, self.model.num_patches)

    @property
    def keep_epochs(self) -> int:
        _, inter = budget_to_mass_mask_ratios(self.budget)
        return visible_count(inter, self.model.num_epoches)
