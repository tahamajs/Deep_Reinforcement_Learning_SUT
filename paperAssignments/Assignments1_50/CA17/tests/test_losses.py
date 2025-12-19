import torch

from paperAssignments.Assignments1_50.CA17.src.losses import policy_gradient_loss, entropy_loss


def test_policy_gradient_loss_zero_advantage():
    logp = torch.tensor([0.0, -0.5, 0.2])
    adv = torch.zeros_like(logp)
    loss = policy_gradient_loss(logp, adv)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_entropy_loss_positive():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ent = entropy_loss(logits, coeff=0.1)
    # entropy_loss returns -coeff * entropy; entropy is positive so loss should be negative
    assert ent < 0.0
