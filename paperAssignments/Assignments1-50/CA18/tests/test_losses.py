import torch
from losses import compute_gae, policy_value_losses


def test_gae_shape_and_values():
    # B=2, T=4
    rewards = torch.tensor([[1.0, 0.0, 0.0, 2.0], [0.5, 0.5, 0.0, 0.0]])
    values = torch.zeros(2, 5)  # includes bootstrap value
    dones = torch.zeros(2, 4)
    adv = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (2, 4)
    # advantage values should be finite
    assert torch.isfinite(adv).all()


def test_policy_value_losses_shapes():
    B = 3
    logp = torch.randn(B)
    advantages = torch.randn(B)
    values = torch.randn(B)
    returns = torch.randn(B)
    out = policy_value_losses(logp, advantages, values, returns)
    assert set(out.keys()) == {"policy_loss", "value_loss", "total"}
    assert out["policy_loss"].ndim == 0
    assert out["value_loss"].ndim == 0
    assert out["total"].ndim == 0
