from pathlib import Path
import sys

# Ensure CA15/src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "CA15" / "src"
sys.path.insert(0, str(SRC))

from train import train  # noqa: E402


def test_train_callable():
    assert callable(train)
