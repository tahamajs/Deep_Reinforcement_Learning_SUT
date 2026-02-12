"""Pytest configuration for CA28 assignment.

Adds the local `src` directory to `sys.path` so tests can import the package
when run from the repository root.
"""

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

src_str = str(_SRC)
if _SRC.exists() and src_str not in sys.path:
    sys.path.insert(0, src_str)
