"""Pytest configuration for CA23 assignment.

Ensures the sibling `src` package is importable when pytest is launched from
the repository root (tests import `src.*`).
"""

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

src_str = str(_SRC)
if _SRC.exists() and src_str not in sys.path:
    sys.path.insert(0, src_str)
