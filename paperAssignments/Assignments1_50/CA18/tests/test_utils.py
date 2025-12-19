import torch
from src.utils import set_seed


def test_set_seed_reproducible():
    set_seed(12345)
    a = torch.randn(4)
    set_seed(12345)
    b = torch.randn(4)
    assert torch.allclose(a, b)
