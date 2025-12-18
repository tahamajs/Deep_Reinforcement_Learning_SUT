"""Tests for utils module."""
import torch
import numpy as np
from src.utils import set_seed


def test_set_seed():
    """Test setting random seed."""
    set_seed(42)
    a = np.random.rand()
    b = torch.rand(1).item()

    set_seed(42)
    c = np.random.rand()
    d = torch.rand(1).item()

    assert a == c
    assert b == d