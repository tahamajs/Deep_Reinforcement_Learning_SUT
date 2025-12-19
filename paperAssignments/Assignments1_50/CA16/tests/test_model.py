import torch
import numpy as np
from paperAssignments.Assignments1_50.CA16.src.model import MLPPolicy, MLPValue
from paperAssignments.Assignments1_50.CA16.src.losses import policy_loss, value_loss
from paperAssignments.Assignments1_50.CA16.src.data import ReplayBuffer
from paperAssignments.Assignments1_50.CA16.src.utils import set_seed, to_tensor


def test_policy_and_value_forward():
    obs_dim = 4
    action_dim = 2
    batch = 8
    obs = torch.randn(batch, obs_dim)

    policy = MLPPolicy(obs_dim, action_dim, hidden_dim=32)
    value = MLPValue(obs_dim, hidden_dim=32)

    logits = policy(obs)
    assert logits.shape == (batch, action_dim)

    actions, logp = policy.get_action(obs)
    assert actions.shape[0] == batch
    assert logp.shape[0] == batch

    vals = value(obs)
    assert vals.shape == (batch,)

    # synthetic advantages/targets
    advantages = torch.randn(batch)
    pg_loss = policy_loss(logp, advantages)
    assert pg_loss.shape == ()

    targets = torch.randn(batch)
    v_loss = value_loss(vals, targets)
    assert v_loss.shape == ()


def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        obs = np.random.randn(4)
        buf.add(obs, int(i % 2), float(i), obs * 0.1, False)
    assert len(buf) == 5

    obs_b, actions, rewards, next_obs, dones = buf.sample(3)
    assert obs_b.shape[0] == 3
    assert actions.shape[0] == 3
    assert rewards.shape[0] == 3
    assert next_obs.shape[0] == 3
    assert dones.shape[0] == 3


def test_utils_set_seed_and_to_tensor():
    set_seed(123)
    a = to_tensor([1, 2, 3])
    assert a.dtype in (torch.int64, torch.int32)
    b = to_tensor(a, device="cpu")
    assert b.device.type == "cpu"


def test_replay_buffer_sample_too_large_raises():
    import pytest

    buf = ReplayBuffer(capacity=5)
    for i in range(3):
        buf.add(np.zeros(4), 0, 0.0, np.zeros(4), False)

    with pytest.raises(ValueError):
        buf.sample(10)


def test_count_parameters_nonzero():
    from paperAssignments.Assignments1_50.CA16.src.utils import count_parameters

    policy = MLPPolicy(4, 2)
    assert count_parameters(policy) > 0















