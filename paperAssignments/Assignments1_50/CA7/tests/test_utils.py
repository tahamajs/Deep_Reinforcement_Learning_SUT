import torch
from src.utils import compute_staleness


def test_compute_staleness_basic():
    # two identical hidden vectors -> staleness 0
    a = torch.randn(4, 16)
    s = compute_staleness(a, a)
    assert s.shape == (4,)
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-6)

    # opposite vectors -> staleness near 2
    b = -a
    s2 = compute_staleness(a, b)
    assert s2.shape == (4,)
    assert (s2 > 1.9).all()
