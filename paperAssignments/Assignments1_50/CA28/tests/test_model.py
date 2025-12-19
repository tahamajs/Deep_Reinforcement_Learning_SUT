import torch
from src.model import QNetwork

def test_q_network():
    """Test QNetwork forward pass."""
    model = QNetwork(4, 2)
    state = torch.randn(1, 4)
    q_values = model(state)
    assert q_values.shape == (1, 2)