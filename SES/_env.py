"""Windows OpenMP / PyTorch DLL bootstrap. Import before numpy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    prefix = Path(sys.prefix)
    for extra in (prefix / "Library" / "bin", prefix / "Lib" / "site-packages" / "torch" / "lib"):
        if extra.is_dir():
            try:
                os.add_dll_directory(str(extra))
            except OSError:
                pass
            os.environ["PATH"] = str(extra) + os.pathsep + os.environ.get("PATH", "")

import torch as _torch  # noqa: E402, F401
