from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grad_rl.algorithms.policy.common import CategoricalActor, GaussianActor, compute_gae, minibatches
from grad_rl.core import RunLogger, evaluate_agent, set_seed
from grad_rl.core.networks import ValueNet
from grad_rl.envs import make_env


@dataclass
class PPOConfig:
    env: str = "CartPole-v1"
    total_steps: int = 40000
    horizon: int = 1024
    epochs: int = 8
    batch_size: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_sizes: tuple = (128, 128)


class PPOAgent:
    def __init__(self, obs_dim: int, action_space, cfg: PPOConfig, device):
        self.device = device
        self.is_discrete = hasattr(action_space, "n")
        if self.is_discrete:
            self.actor = CategoricalActor(obs_dim, action_space.n, hidden=cfg.hidden_sizes).to(device)
        else:
            self.actor = GaussianActor(obs_dim, int(np.prod(action_space.shape)), hidden=cfg.hidden_sizes).to(device)
            self.act_low = torch.tensor(action_space.low, dtype=torch.float32, device=device)
            self.act_high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
        self.critic = ValueNet(obs_dim, hidden=cfg.hidden_sizes).to(device)
        self.optim = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.learning_rate)
        self.cfg = cfg

    def _dist(self, obs_t):
        if self.is_discrete:
            return self.actor(obs_t)
        return self.actor.dist(obs_t)

    def act(self, obs, deterministic: bool = False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self._dist(obs_t)
        if self.is_discrete:
            if deterministic:
                return int(torch.argmax(dist.logits, dim=1).item())
            return int(dist.sample().item())
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        action = torch.clamp(action, self.act_low, self.act_high)
        return action.squeeze(0).detach().cpu().numpy()


def train_ppo(config: Dict, out_dir: Path, seed: int, algo_name: str = "ppo") -> Dict:
    cfg = PPOConfig(**{k: v for k, v in config.items() if k in PPOConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = PPOAgent(obs_dim, env.action_space, cfg, device)

    logger = RunLogger(
        run_id=f"{algo_name}_{cfg.env.replace('/', '_')}_s{seed}",
        chain="policy",
        algo=algo_name,
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    obs, _ = env.reset()
    step_count = 0

    while step_count < cfg.total_steps:
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        for _ in range(cfg.horizon):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = agent._dist(obs_t)
            value = agent.critic(obs_t).item()
            if agent.is_discrete:
                action_t = dist.sample()
                action = int(action_t.item())
                logp = dist.log_prob(action_t).item()
            else:
                action_t = dist.sample()
                logp = dist.log_prob(action_t).sum(dim=-1).item()
                action = torch.clamp(action_t, agent.act_low, agent.act_high).squeeze(0).detach().cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs.copy())
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            val_buf.append(float(value))

            obs = next_obs
            step_count += 1
            if done:
                logger.log_train(step_count, sum(rew_buf[-20:]))
                obs, _ = env.reset()
            if step_count >= cfg.total_steps:
                break

        returns, adv = compute_gae(rew_buf, val_buf, done_buf, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        if agent.is_discrete:
            act_t = torch.tensor(np.array(act_buf), dtype=torch.int64, device=device)
        else:
            act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        for _ in range(cfg.epochs):
            for mb_idx in minibatches(len(obs_buf), cfg.batch_size):
                dist = agent._dist(obs_t[mb_idx])
                if agent.is_discrete:
                    logp = dist.log_prob(act_t[mb_idx])
                    entropy = dist.entropy().mean()
                else:
                    logp = dist.log_prob(act_t[mb_idx]).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(logp - logp_old_t[mb_idx])
                surr1 = ratio * adv_t[mb_idx]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                values = agent.critic(obs_t[mb_idx]).squeeze(1)
                value_loss = ((ret_t[mb_idx] - values) ** 2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                agent.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(agent.actor.parameters()) + list(agent.critic.parameters()), cfg.max_grad_norm)
                agent.optim.step()

    eval_stats = evaluate_agent(agent, env, episodes=5, deterministic=True)
    payload = logger.finalize(eval_stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, out_dir / "model.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
