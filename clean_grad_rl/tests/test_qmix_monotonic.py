import torch

from grad_rl.core.networks import MonotonicMixer


def test_qmix_mixer_monotonic_numerical_check():
    mixer = MonotonicMixer(n_agents=3, state_dim=9, hidden_dim=8)
    state = torch.randn(5, 9)
    q = torch.randn(5, 3)
    out1 = mixer(q, state)
    q2 = q.clone()
    q2[:, 1] += 0.5
    out2 = mixer(q2, state)
    assert torch.all(out2 >= out1 - 1e-5)
