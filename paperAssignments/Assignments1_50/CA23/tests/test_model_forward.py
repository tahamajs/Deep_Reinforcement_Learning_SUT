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

    # single observation should also work and return squeezed tensors/scalars
    a_single, lp_single = policy.get_action(x[0])
    assert (isinstance(a_single, torch.Tensor) and a_single.ndim == 0) or isinstance(a_single, int)
    assert (isinstance(lp_single, torch.Tensor) and lp_single.ndim == 0) or isinstance(lp_single, float)

    v = value(x)
    assert v.shape == (batch,)
