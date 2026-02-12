import numpy as np

from grad_rl.core.buffers import ReplayBuffer


def test_prioritized_replay_sampling_and_update():
    rb = ReplayBuffer(capacity=100, obs_shape=(4,), prioritized=True, alpha=0.6)
    for i in range(80):
        obs = np.ones(4, dtype=np.float32) * i
        rb.add(obs, int(i % 2), float(i), obs + 1.0, 0.0)

    batch = rb.sample(batch_size=16, beta=0.4)
    assert batch["obs"].shape == (16, 4)
    assert batch["weights"].shape == (16,)

    td = np.linspace(0.1, 1.6, num=16)
    rb.update_priorities(batch["indices"], td)
    assert np.all(rb.priorities[batch["indices"]] > 0)
