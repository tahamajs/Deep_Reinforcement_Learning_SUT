import numpy as np
from src.prioritized_replay import PrioritizedReplayBuffer


def test_prioritized_push_and_sample():
    buf = PrioritizedReplayBuffer(10, alpha=0.6)
    state = np.zeros(4)
    for i in range(6):
        buf.push(state + i, i % 2, float(i), state + i + 1, False)
    assert len(buf) == 6

    states, actions, rewards, next_states, dones, indices, weights = buf.sample(4)
    assert states.shape == (4, 4)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 4)
    assert dones.shape == (4,)
    assert indices.shape == (4,)
    assert weights.shape == (4,)

    # Update priorities - should not raise
    errors = np.random.rand(4)
    buf.update_priorities(indices, errors)
    # priorities at indices should be > 0
    assert np.all(buf.priorities[indices] > 0)
