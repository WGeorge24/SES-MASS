"""SES: equal-interval whole-epoch retention for wearable sleep staging.

Whole 30 s epochs use the same equal-interval rule as the 1 s grid instead
of being redrawn every night. Deployment phase is 0, so the on/standby
timetable and the collected seconds are fixed.

    python -m SES --schedule
    python -m SES --smoke
    python -m SES --dataset dreams --budget 0.72
"""

from . import _env  # noqa: F401
from .paths import ROOT, PACKAGE_DIR
from .config import DATASETS, PI_CHOICES, BUDGET_GRID

__all__ = ["ROOT", "PACKAGE_DIR", "DATASETS", "PI_CHOICES", "BUDGET_GRID"]
