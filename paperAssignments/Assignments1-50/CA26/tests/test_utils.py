from pathlib import Path

import numpy as np
import torch

from src.utils import set_seed, ensure_dir, save_loss_curve


def test_set_seed_reproducible():
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert (a == b).all()
    # Torch sequence
    set_seed(456)
    t1 = torch.rand(3)
    set_seed(456)
    t2 = torch.rand(3)
    assert torch.allclose(t1, t2)


def test_ensure_dir_and_save_loss_curve(tmp_path: Path):
    out_dir = tmp_path / "out_dir"
    p = ensure_dir(out_dir)
    assert p.exists() and p.is_dir()
    losses = [1.0, 0.8, 0.6]
    save_loss_curve(losses, p / "loss")
    img = p / "loss" / "loss_curve.png"
    assert img.exists()
