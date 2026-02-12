from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grad_rl.algorithms.actor_critic.sac import SACAgent, SACConfig
from grad_rl.core import ReplayBuffer, RunLogger, evaluate_agent, set_seed
from grad_rl.envs import make_env


class DynamicsModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim + 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        out = self.net(x)
        delta = out[:, :-1]
        reward = out[:, -1:]
        return delta, reward


@dataclass
class MBPOConfig:
    env: str = "Pendulum-v1"
    total_steps: int = 50000
    ensemble_size: int = 5
    model_batch_size: int = 256
    model_updates_per_step: int = 1
    rollout_horizon: int = 3
    rollout_batch_size: int = 128
    real_ratio: float = 0.5
    learning_starts: int = 1000
    hidden_size: int = 256


def train_mbpo_lite(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = MBPOConfig(**{k: v for k, v in config.items() if k in MBPOConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg.env, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    sac_cfg = SACConfig(env=cfg.env, total_steps=cfg.total_steps)
    agent = SACAgent(obs_dim, act_dim, env.action_space.low, env.action_space.high, sac_cfg, device)

    real_rb = ReplayBuffer(100000, obs_shape=env.observation_space.shape, action_shape=env.action_space.shape)
    model_rb = ReplayBuffer(100000, obs_shape=env.observation_space.shape, action_shape=env.action_space.shape)

    ensemble = [DynamicsModel(obs_dim, act_dim, hidden=cfg.hidden_size).to(device) for _ in range(cfg.ensemble_size)]
    model_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in ensemble]

    logger = RunLogger(
        run_id=f"mbpo_lite_{cfg.env.replace('/', '_')}_s{seed}",
        chain="model_based",
        algo="mbpo_lite",
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    model_losses: List[float] = []

    for step in range(1, cfg.total_steps + 1):
        if step < cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        real_rb.add(obs, action, float(reward), next_obs, float(done))
        obs = next_obs
        ep_reward += float(reward)

        if done:
            logger.log_train(step, ep_reward)
            obs, _ = env.reset()
            ep_reward = 0.0

        if len(real_rb) >= max(cfg.model_batch_size, cfg.learning_starts):
            for _ in range(cfg.model_updates_per_step):
                batch = real_rb.sample(cfg.model_batch_size)
                o = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                a = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
                n = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                r = torch.tensor(batch["rewards"], dtype=torch.float32, device=device).unsqueeze(1)
                delta_target = n - o

                for m, opt in zip(ensemble, model_opts):
                    delta_pred, r_pred = m(o, a)
                    loss = ((delta_pred - delta_target) ** 2).mean() + ((r_pred - r) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    model_losses.append(float(loss.item()))

            # short model rollouts from real states
            seed_batch = real_rb.sample(cfg.rollout_batch_size)
            mo = torch.tensor(seed_batch["obs"], dtype=torch.float32, device=device)
            for _ in range(cfg.rollout_horizon):
                with torch.no_grad():
                    ma = torch.tensor(
                        np.stack([agent.act(x.cpu().numpy(), deterministic=False) for x in mo]),
                        dtype=torch.float32,
                        device=device,
                    )
                    preds = [m(mo, ma) for m in ensemble]
                    idx = np.random.randint(cfg.ensemble_size)
                    delta, rew = preds[idx]
                    mn = mo + delta
                    done_prob = torch.zeros((mo.shape[0], 1), device=device)

                for i in range(mo.shape[0]):
                    model_rb.add(
                        mo[i].cpu().numpy(),
                        ma[i].cpu().numpy(),
                        float(rew[i].item()),
                        mn[i].cpu().numpy(),
                        float(done_prob[i].item()),
                    )
                mo = mn

        if step > cfg.learning_starts and len(real_rb) >= 256:
            total_batch = 256
            real_n = int(total_batch * cfg.real_ratio)
            model_n = total_batch - real_n
            rb1 = real_rb.sample(real_n)
            if len(model_rb) >= model_n and model_n > 0:
                rb2 = model_rb.sample(model_n)
                batch = {
                    k: np.concatenate([rb1[k], rb2[k]], axis=0)
                    for k in ["obs", "actions", "rewards", "next_obs", "dones"]
                }
            else:
                batch = {k: rb1[k] for k in ["obs", "actions", "rewards", "next_obs", "dones"]}
                batch["weights"] = np.ones((len(batch["obs"]),), dtype=np.float32)
                batch["indices"] = np.arange(len(batch["obs"]))
            if "weights" not in batch:
                batch["weights"] = np.ones((len(batch["obs"]),), dtype=np.float32)
                batch["indices"] = np.arange(len(batch["obs"]))
            agent.train_step(batch)

    eval_stats = evaluate_agent(agent, env, episodes=5, deterministic=True)
    payload = logger.finalize(
        eval_stats,
        extra={"train_summary": {"mean_model_loss": float(np.mean(model_losses)) if model_losses else 0.0}},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "dynamics": [m.state_dict() for m in ensemble],
        },
        out_dir / "model.pt",
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
