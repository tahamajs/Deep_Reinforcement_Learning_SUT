from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from grad_rl.algorithms.policy.common import CategoricalActor
from grad_rl.core import RunLogger, evaluate_agent, set_seed
from grad_rl.envs import make_env


@dataclass
class ReinforceConfig:
    env: str = "CartPole-v1"
    total_steps: int = 30000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_sizes: tuple = (128, 128)


class ReinforceAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: ReinforceConfig, device):
        self.actor = CategoricalActor(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.optim = optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.device = device

    def act(self, obs, deterministic: bool = False):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(x)
        if deterministic:
            return int(torch.argmax(dist.logits, dim=1).item())
        return int(dist.sample().item())


def train_reinforce(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = ReinforceConfig(**{k: v for k, v in config.items() if k in ReinforceConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n
    agent = ReinforceAgent(obs_dim, act_dim, cfg, device)

    logger = RunLogger(
        run_id=f"reinforce_{cfg.env.replace('/', '_')}_s{seed}",
        chain="policy",
        algo="reinforce",
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    step_count = 0
    episode = 0
    while step_count < cfg.total_steps:
        obs, _ = env.reset()
        done = False
        logps = []
        rewards = []
        ep_reward = 0.0
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = agent.actor(x)
            action = dist.sample()
            logps.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rewards.append(float(reward))
            ep_reward += float(reward)
            step_count += 1
            if step_count >= cfg.total_steps:
                break

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + cfg.gamma * G
            returns.append(G)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -torch.sum(torch.stack(logps) * returns)
        agent.optim.zero_grad()
        loss.backward()
        agent.optim.step()

        episode += 1
        logger.log_train(step_count, ep_reward)

    eval_stats = evaluate_agent(agent, env, episodes=5, deterministic=True)
    payload = logger.finalize(eval_stats, extra={"train_summary": {"episodes": episode}})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), out_dir / "model.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
