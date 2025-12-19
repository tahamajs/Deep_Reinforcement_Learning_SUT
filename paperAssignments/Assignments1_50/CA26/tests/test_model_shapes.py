import torch

from src.model import MLP


def test_mlp_forward_shape():
    model = MLP(input_dim=3, hidden_dims=(8, 8), output_dim=2)
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 2)
