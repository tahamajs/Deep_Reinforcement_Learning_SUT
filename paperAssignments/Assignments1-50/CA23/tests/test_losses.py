import pytest
pytest.importorskip("torch")
import torch
from src.losses import policy_gradient_loss, value_loss, entropy_loss


def test_losses_shapes_and_values():
    logp = torch.tensor([0.0, -0.5, -1.0])
    adv = torch.tensor([1.0, 0.5, -0.2])
    pg = policy_gradient_loss(logp, adv)
    assert pg.shape == ()

    vals = torch.tensor([1.0, 2.0, 3.0])
    rets = torch.tensor([1.1, 1.9, 2.6])
    vl = value_loss(vals, rets)
    assert vl.item() >= 0.0

    probs = torch.tensor([[0.5, 0.5], [0.8, 0.2]])
    ent = entropy_loss(probs)
    assert ent.shape == ()
