import pytest
pytest.importorskip("torch")
import torch
from src.model import PolicyNetwork, ValueNetwork


def test_policy_and_value_forward_shapes():
    obs_dim = 4
    action_dim = 2
    batch = 3
    policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes=(16, 16))
    value = ValueNetwork(obs_dim, hidden_sizes=(16, 16))

    x = torch.randn(batch, obs_dim)
    logits = policy(x)
    assert logits.shape == (batch, action_dim)

    actions, logp = policy.get_action(x)
    assert actions.shape[0] == batch
    assert logp.shape[0] == batch

    v = value(x)
    assert v.shape == (batch,)
