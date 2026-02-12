import numpy as np
import torch

from grad_rl.algorithms.actor_critic.sac import SACAgent, SACConfig


def test_sac_alpha_stays_finite_after_update():
    obs_dim = 3
    act_dim = 1
    cfg = SACConfig(total_steps=10, batch_size=4)
    device = torch.device("cpu")
    agent = SACAgent(obs_dim, act_dim, np.array([-1.0]), np.array([1.0]), cfg, device)

    batch = {
        "obs": np.random.randn(4, obs_dim).astype(np.float32),
        "actions": np.random.uniform(-1, 1, size=(4, act_dim)).astype(np.float32),
        "rewards": np.random.randn(4).astype(np.float32),
        "next_obs": np.random.randn(4, obs_dim).astype(np.float32),
        "dones": np.zeros(4, dtype=np.float32),
        "weights": np.ones(4, dtype=np.float32),
        "indices": np.arange(4),
    }
    stats = agent.train_step(batch)
    assert np.isfinite(stats["alpha"])
    assert stats["alpha"] > 0
