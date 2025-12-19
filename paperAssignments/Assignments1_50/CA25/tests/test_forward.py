import torch
from src.model import MLP


def test_forward_shapes():
    m = MLP(input_dim=8, hidden_dims=(16, 16), output_dim=1)
    x = torch.randn(4, 8)
    out = m(x)
    assert out.shape == (4, 1)

    m2 = MLP(input_dim=8, hidden_dims=(8,), output_dim=3)
    out2 = m2(x)
    assert out2.shape == (4, 3)
