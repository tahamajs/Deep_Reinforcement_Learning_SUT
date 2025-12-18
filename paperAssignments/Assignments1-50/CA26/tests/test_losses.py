import torch

from src.losses import mse_loss, huber_loss


def test_mse_basic():
    a = torch.tensor([[1.0], [2.0]])
    b = torch.tensor([[1.0], [3.0]])
    loss = mse_loss(a, b)
    assert torch.isclose(loss, torch.tensor(0.25))


def test_mse_shape_mismatch_raises():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([[1.0], [2.0]])
    try:
        _ = mse_loss(a, b)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_huber_limits():
    a = torch.tensor([0.0, 10.0])
    b = torch.tensor([0.0, 0.0])
    # With large delta it becomes MSE
    h = huber_loss(a, b, delta=100.0)
    m = mse_loss(a, b)
    assert torch.isclose(h, m)
