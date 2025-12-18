import pytest
import torch
from src.model import MLPPolicy
from src.utils import set_seed, discounted_returns, returns_to_tensor
from src.losses import reinforce_loss


def test_model_forward_and_action():
    set_seed(0)
    obs_dim = 4
    action_dim = 2
    model = MLPPolicy(obs_dim, action_dim)
    obs = torch.randn(obs_dim)
    logits = model.forward(obs)
    assert logits.shape[-1] == action_dim
    a, logp = model.get_action(obs)
    assert isinstance(a, torch.Tensor) or isinstance(a, int)


def test_discounted_returns_and_loss():
    rewards = [1.0, 0.0, 2.0]
    gamma = 0.9
    G = discounted_returns(rewards, gamma)
    assert len(G) == len(rewards)
    Gt = returns_to_tensor(G)
    # fake log_probs
    log_probs = torch.tensor([0.0, -0.1, -0.2], dtype=torch.float32)
    loss = reinforce_loss(log_probs, Gt)
    assert torch.isfinite(loss)


