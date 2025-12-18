from paperAssignments.Assignments1_50.CA17.src.utils import set_seed, ensure_dir
from pathlib import Path


def test_ensure_dir(tmp_path):
    p = tmp_path / "outputs_test"
    assert ensure_dir(p) == p
    assert p.exists()


def test_set_seed_deterministic():
    set_seed(123)
    import numpy as np

    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert (a == b).all()
