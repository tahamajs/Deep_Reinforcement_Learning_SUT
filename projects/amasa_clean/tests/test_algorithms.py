import numpy as np
import torch

from projects.amasa_clean.amasa.offline import CQLAgent, CQLConfig, IQLAgent, IQLConfig
from projects.amasa_clean.amasa.online import SACLagrangianAgent, SACLagConfig, PPOLagrangianAgent, PPOLagConfig


OBS_DIM = 23
ACT_DIM = 7


def _batch(batch_size=16):
    obs = torch.randn(batch_size, OBS_DIM)
    act = torch.tanh(torch.randn(batch_size, ACT_DIM))
    rew = torch.randn(batch_size, 1)
    nxt = torch.randn(batch_size, OBS_DIM)
    done = torch.zeros(batch_size, 1)
    cost = torch.rand(batch_size, 1)
    return obs, act, rew, nxt, done, cost


def test_cql_update_step():
    agent = CQLAgent(CQLConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM, device="cpu"))
    obs, act, rew, nxt, done, _ = _batch()
    m = agent.update((obs, act, rew, nxt, done))
    assert "critic_loss" in m


def test_iql_update_step():
    agent = IQLAgent(IQLConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM, device="cpu"))
    obs, act, rew, nxt, done, _ = _batch()
    m = agent.update((obs, act, rew, nxt, done))
    assert "q_loss" in m


def test_sac_lag_update_step():
    agent = SACLagrangianAgent(SACLagConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM, device="cpu"))
    obs, act, rew, nxt, done, cost = _batch()
    m = agent.update((obs, act, rew, nxt, done, cost), lambda_value=1.0)
    assert "actor_loss" in m


def test_ppo_lag_update_step():
    agent = PPOLagrangianAgent(PPOLagConfig(obs_dim=OBS_DIM, act_dim=ACT_DIM, device="cpu"))
    n = 32
    traj = {
        "obs": np.random.randn(n, OBS_DIM).astype(np.float32),
        "act": np.tanh(np.random.randn(n, ACT_DIM)).astype(np.float32),
        "logp": np.random.randn(n).astype(np.float32),
        "rew": np.random.randn(n).astype(np.float32),
        "cost": np.random.rand(n).astype(np.float32),
        "done": np.zeros(n, dtype=np.float32),
    }
    built = agent.build_trajectory(traj)
    m = agent.update(built, lambda_value=1.0)
    assert "policy_loss" in m
