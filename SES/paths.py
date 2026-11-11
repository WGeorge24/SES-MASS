"""Ensure the MASS-main repo root is importable (utils, structure, lion_pytorch)."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_LGSLEEP = ROOT / "others" / "LGSleepNet"
if _LGSLEEP.exists() and str(_LGSLEEP) not in sys.path:
    sys.path.insert(0, str(_LGSLEEP))
