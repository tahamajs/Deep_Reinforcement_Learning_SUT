from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grad_rl.core import ReplayBuffer, RunLogger, evaluate_agent, set_seed
from grad_rl.core.networks import ActorGaussian, CriticQ
from grad_rl.envs import make_env


@dataclass
class SACConfig:
    env: str = "Pendulum-v1"
    total_steps: int = 60000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100000
    learning_starts: int = 1000
    train_freq: int = 1
    alpha_lr: float = 3e-4
    target_entropy_scale: float = 1.0
    hidden_sizes: tuple = (256, 256)


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high, cfg: SACConfig, device):
        self.actor = ActorGaussian(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.q1 = CriticQ(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.q2 = CriticQ(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.q1_t = CriticQ(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.q2_t = CriticQ(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.learning_rate)

        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = -float(act_dim) * cfg.target_entropy_scale

        self.cfg = cfg
        self.device = device
        self.act_low = torch.tensor(act_low, dtype=torch.float32, device=device)
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _sample_action(self, obs_t):
        mu, log_std = self.actor(obs_t)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        scaled = self.act_low + (action + 1.0) * 0.5 * (self.act_high - self.act_low)
        # tanh correction
        logp = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return scaled, logp.sum(dim=-1, keepdim=True)

    def act(self, obs, deterministic: bool = False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, log_std = self.actor(obs_t)
            if deterministic:
                z = mu
            else:
                z = mu + log_std.exp() * torch.randn_like(mu)
            action = torch.tanh(z)
            scaled = self.act_low + (action + 1.0) * 0.5 * (self.act_high - self.act_low)
        return scaled.squeeze(0).cpu().numpy()

    def train_step(self, batch):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_a, next_logp = self._sample_action(next_obs)
            q1_next = self.q1_t(next_obs, next_a)
            q2_next = self.q2_t(next_obs, next_a)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            target = rewards + (1.0 - dones) * self.cfg.gamma * q_next

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q_loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        new_a, logp = self._sample_action(obs)
        q_pi = torch.min(self.q1(obs, new_a), self.q2(obs, new_a))
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }


def train_sac(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = SACConfig(**{k: v for k, v in config.items() if k in SACConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    agent = SACAgent(obs_dim, act_dim, env.action_space.low, env.action_space.high, cfg, device)
    rb = ReplayBuffer(cfg.buffer_size, obs_shape=env.observation_space.shape, action_shape=env.action_space.shape)

    logger = RunLogger(
        run_id=f"sac_{cfg.env.replace('/', '_')}_s{seed}",
        chain="actor_critic",
        algo="sac",
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    losses = []

    for step in range(1, cfg.total_steps + 1):
        if step < cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rb.add(obs, action, float(reward), next_obs, float(done))
        obs = next_obs
        ep_reward += float(reward)

        if step > cfg.learning_starts and step % cfg.train_freq == 0 and len(rb) >= cfg.batch_size:
            batch = rb.sample(cfg.batch_size)
            losses.append(agent.train_step(batch))

        if done:
            logger.log_train(step, ep_reward, cost=float(info.get("cost", 0.0)) if isinstance(info, dict) else None)
            obs, _ = env.reset()
            ep_reward = 0.0

    eval_stats = evaluate_agent(agent, env, episodes=5, deterministic=True)
    extra = {
        "train_summary": {
            "mean_q_loss": float(np.mean([x["q_loss"] for x in losses])) if losses else 0.0,
            "mean_actor_loss": float(np.mean([x["actor_loss"] for x in losses])) if losses else 0.0,
            "final_alpha": float(losses[-1]["alpha"]) if losses else float(agent.alpha.item()),
        }
    }
    payload = logger.finalize(eval_stats, extra=extra)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "log_alpha": float(agent.log_alpha.detach().cpu().item()),
        },
        out_dir / "model.pt",
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
