import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ca19.utils", str(Path(__file__).resolve().parent.parent / "src" / "utils.py"))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)
set_seed = mod.set_seed
save_checkpoint = mod.save_checkpoint
load_checkpoint = mod.load_checkpoint


def test_set_seed_deterministic():
    import numpy as np
    import torch

    set_seed(12345)
    a1 = torch.randn(4)
    b1 = np.random.randn(4)

    set_seed(12345)
    a2 = torch.randn(4)
    b2 = np.random.randn(4)

    assert torch.allclose(a1, a2)
    assert (b1 == b2).all()


def test_checkpoint_save_load(tmp_path):
    d = tmp_path / "ckpts"
    d.mkdir()
    p = d / "test_ckpt.pt"
    state = {"a": 1, "tensor": __import__("torch").tensor([1, 2, 3])}
    save_checkpoint(state, str(p))
    loaded = load_checkpoint(str(p), device=__import__("torch").device("cpu"))
    assert loaded["a"] == 1
    assert __import__("torch").allclose(loaded["tensor"], state["tensor"])