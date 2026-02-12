import numpy as np

from grad_rl.core.buffers import NStepAccumulator, Transition


def test_nstep_return_computation():
    nstep = NStepAccumulator(n_step=3, gamma=0.9)
    obs = np.zeros(2, dtype=np.float32)
    nstep.push(Transition(obs, 0, 1.0, obs, 0.0))
    nstep.push(Transition(obs, 0, 2.0, obs, 0.0))
    nstep.push(Transition(obs, 0, 3.0, obs, 0.0))
    assert nstep.ready()
    _, _, r, _, _ = nstep.pop_nstep()
    expected = 1.0 + 0.9 * 2.0 + (0.9 ** 2) * 3.0
    assert abs(r - expected) < 1e-6
