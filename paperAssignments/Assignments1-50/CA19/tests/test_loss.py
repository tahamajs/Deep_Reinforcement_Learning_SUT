import torch
from paperAssignments.Assignments1_50.CA19.src.losses import actor_loss, critic_loss, value_ensemble_variance


def test_losses_finite():
    B = 6
    A = 3
    logits = torch.randn(B, A)
    actions = torch.randint(0, A, (B,))
    advantages = torch.randn(B)
    values = torch.randn(3, B)
    targets = torch.randn(B)
    var = value_ensemble_variance(values)
    a_loss = actor_loss(logits, actions, advantages, var, beta=0.1)
    c_loss = critic_loss(values, targets)
    assert torch.isfinite(a_loss)
    assert torch.isfinite(c_loss)
