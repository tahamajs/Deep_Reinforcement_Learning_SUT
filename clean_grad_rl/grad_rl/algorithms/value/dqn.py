from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grad_rl.core import NStepAccumulator, ReplayBuffer, RunLogger, Transition, evaluate_agent, linear_schedule, set_seed
from grad_rl.core.networks import DuelingQNet, mlp, NoisyLinear
from grad_rl.envs import make_env


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), noisy: bool = False):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim, noisy=noisy)

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    env: str = "CartPole-v1"
    total_steps: int = 30000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50000
    learning_starts: int = 1000
    target_update_interval: int = 500
    exploration_start: float = 1.0
    exploration_final_eps: float = 0.02
    exploration_fraction: float = 0.1
    train_freq: int = 1
    grad_clip_norm: float = 10.0
    hidden_sizes: Tuple[int, int] = (128, 128)
    double: bool = True
    dueling: bool = True
    prioritized_replay: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 30000
    n_step: int = 1
    noisy: bool = False


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DQNConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        if cfg.dueling:
            self.q_net = DuelingQNet(obs_dim, act_dim, hidden=cfg.hidden_sizes, noisy=cfg.noisy).to(device)
            self.target_net = DuelingQNet(obs_dim, act_dim, hidden=cfg.hidden_sizes, noisy=cfg.noisy).to(device)
        else:
            self.q_net = QNet(obs_dim, act_dim, hidden=cfg.hidden_sizes, noisy=cfg.noisy).to(device)
            self.target_net = QNet(obs_dim, act_dim, hidden=cfg.hidden_sizes, noisy=cfg.noisy).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.act_dim = act_dim
        self.gamma_n = cfg.gamma ** cfg.n_step

    def act(self, obs, epsilon: float = 0.0, deterministic: bool = False):
        if not deterministic and np.random.rand() < epsilon:
            return int(np.random.randint(self.act_dim))
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x)
            return int(q.argmax(dim=1).item())

    def train_step(self, batch, beta: float):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(batch["weights"], dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.q_net(obs).gather(1, actions)
        with torch.no_grad():
            if self.cfg.double:
                next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions)
            else:
                next_q = self.target_net(next_obs).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma_n * next_q

        td_error = (q - target).detach().squeeze(1).cpu().numpy()
        loss = (weights * F.smooth_l1_loss(q, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        if self.cfg.noisy:
            for m in self.q_net.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
            for m in self.target_net.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

        return float(loss.item()), td_error


def train_dqn_like(config: Dict, out_dir: Path, seed: int, algo_name: str = "dqn") -> Dict:
    cfg = DQNConfig(**{k: v for k, v in config.items() if k in DQNConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    agent = DQNAgent(obs_dim, act_dim, cfg, device)
    rb = ReplayBuffer(
        cfg.buffer_size,
        obs_shape=env.observation_space.shape,
        action_shape=(),
        prioritized=cfg.prioritized_replay,
        alpha=cfg.per_alpha,
    )
    nstep = NStepAccumulator(cfg.n_step, cfg.gamma)

    run_id = f"{algo_name}_{cfg.env.replace('/', '_')}_s{seed}"
    logger = RunLogger(
        run_id=run_id,
        chain="value",
        algo=algo_name,
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_idx = 0
    losses = []

    for step in range(1, cfg.total_steps + 1):
        eps = linear_schedule(
            cfg.exploration_start,
            cfg.exploration_final_eps,
            step,
            int(cfg.total_steps * cfg.exploration_fraction),
        )
        action = agent.act(obs, epsilon=eps, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        tr = Transition(obs=obs, action=action, reward=float(reward), next_obs=next_obs, done=float(done))
        if cfg.n_step > 1:
            nstep.push(tr)
            if nstep.ready():
                s0, a0, r_n, s_n, d_n = nstep.pop_nstep()
                rb.add(s0, a0, r_n, s_n, d_n)
        else:
            rb.add(obs, action, reward, next_obs, float(done))

        obs = next_obs
        ep_reward += float(reward)

        if step > cfg.learning_starts and step % cfg.train_freq == 0 and len(rb) >= cfg.batch_size:
            beta = linear_schedule(cfg.per_beta_start, 1.0, step, cfg.per_beta_frames)
            batch = rb.sample(cfg.batch_size, beta=beta)
            loss, td = agent.train_step(batch, beta)
            rb.update_priorities(batch["indices"], td)
            losses.append(loss)

        if step % cfg.target_update_interval == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        if done:
            ep_idx += 1
            logger.log_train(step, ep_reward, cost=float(info.get("cost", 0.0)) if isinstance(info, dict) else None)
            obs, _ = env.reset()
            ep_reward = 0.0

    eval_stats = evaluate_agent(agent, env, episodes=5, deterministic=True)
    extra = {
        "train_summary": {
            "episodes": ep_idx,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }
    }
    payload = logger.finalize(eval_stats, extra=extra)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.q_net.state_dict(), out_dir / "model.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload


def train_dqn(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = dict(config)
    cfg["prioritized_replay"] = False
    cfg["n_step"] = 1
    cfg["noisy"] = False
    return train_dqn_like(cfg, out_dir, seed, algo_name="dqn")


def train_rainbow_lite(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = dict(config)
    cfg.setdefault("prioritized_replay", True)
    cfg.setdefault("n_step", 3)
    cfg.setdefault("double", True)
    cfg.setdefault("dueling", True)
    cfg.setdefault("noisy", False)
    return train_dqn_like(cfg, out_dir, seed, algo_name="rainbow_lite")
