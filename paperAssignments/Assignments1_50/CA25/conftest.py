"""Test configuration for CA25.

Ensures the local `src` package (sibling to the tests) is importable when
tests are executed from the repository root, avoiding `ModuleNotFoundError`
for `src.*` imports.
"""

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

# Prepend once so imports in this assignment resolve to the correct package.
src_str = str(_SRC)
if _SRC.exists() and src_str not in sys.path:
    sys.path.insert(0, src_str)
