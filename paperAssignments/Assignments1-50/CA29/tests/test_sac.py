"""Tests for basic SAC components: ReplayBuffer, Actor, Critic."""

import numpy as np
import torch

from src.sac import ReplayBuffer, Actor, Critic


def test_replay_buffer_add_and_sample():
    size = 100
    state_dim = 3
    action_dim = 2
    buf = ReplayBuffer(size, state_dim, action_dim)

    # Add some transitions
    for i in range(10):
        state = np.ones(state_dim) * i
        action = np.ones(action_dim) * (i + 0.1)
        buf.add(state, action, float(i), state + 0.5, False)

    assert len(buf) == 10

    s, a, r, ns, d = buf.sample(4)
    assert s.shape == (4, state_dim)
    assert a.shape == (4, action_dim)
    # rewards and dones are column vectors
    assert r.shape == (4, 1)
    assert ns.shape == (4, state_dim)
    assert d.shape == (4, 1)


def test_actor_critic_shapes():
    state_dim = 5
    action_dim = 2
    batch = 7

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)

    states = torch.randn(batch, state_dim)
    actions = torch.randn(batch, action_dim)

    mean, log_std = actor(states)
    assert mean.shape == (batch, action_dim)
    assert log_std.shape == (batch, action_dim)

    sampled_action, log_prob = actor.sample(states)
    assert sampled_action.shape == (batch, action_dim)
    assert log_prob.shape == (batch, 1)

    q = critic(states, sampled_action)
    assert q.shape == (batch, 1)


def test_sac_save_load(tmp_path):
    # Small sanity test that save/load works (CPU only)
    state_dim = 4
    action_dim = 2
    from src.config import SACConfig
    cfg = SACConfig()
    device = torch.device('cpu')

    sac = __import__('src.sac', fromlist=['SAC']).SAC(state_dim, action_dim, cfg, device)

    # randomize params a bit
    for p in sac.actor.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)

    p = tmp_path / "sac_test.pth"
    sac.save(str(p))

    sac2 = __import__('src.sac', fromlist=['SAC']).SAC(state_dim, action_dim, cfg, device)
    sac2.load(str(p))

    # Compare actor parameters
    for a, b in zip(sac.actor.parameters(), sac2.actor.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
