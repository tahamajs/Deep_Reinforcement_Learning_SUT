from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grad_rl.core import RunLogger, linear_schedule, set_seed
from grad_rl.core.networks import MonotonicMixer, mlp


@dataclass
class QMIXConfig:
    env: str = "simple_spread_v3"
    total_steps: int = 30000
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50000
    learning_starts: int = 500
    train_freq: int = 1
    target_update_interval: int = 400
    exploration_start: float = 1.0
    exploration_final: float = 0.05
    exploration_fraction: float = 0.3
    hidden_sizes: tuple = (128, 128)
    mixer_hidden: int = 32
    max_cycles: int = 25


class AgentQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128)):
        super().__init__()
        self.q = mlp(obs_dim, hidden, act_dim)

    def forward(self, obs):
        return self.q(obs)


class MultiAgentReplay:
    def __init__(self, capacity: int):
        self.buf: Deque[Dict] = deque(maxlen=capacity)

    def add(self, item: Dict):
        self.buf.append(item)

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        return batch


def _make_env(max_cycles=25):
    from pettingzoo.mpe import simple_spread_v3

    return simple_spread_v3.parallel_env(max_cycles=max_cycles, continuous_actions=False)


def train_qmix_lite(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = QMIXConfig(**{k: v for k, v in config.items() if k in QMIXConfig.__annotations__})
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = _make_env(max_cycles=cfg.max_cycles)
    obs_dict, _ = env.reset(seed=seed)
    agents = list(env.possible_agents)
    n_agents = len(agents)
    obs_dim = len(obs_dict[agents[0]])
    act_dim = env.action_space(agents[0]).n
    state_dim = obs_dim * n_agents

    q_nets = [AgentQ(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device) for _ in range(n_agents)]
    tgt_q_nets = [AgentQ(obs_dim, act_dim, hidden=cfg.hidden_sizes).to(device) for _ in range(n_agents)]
    for q, tq in zip(q_nets, tgt_q_nets):
        tq.load_state_dict(q.state_dict())

    mixer = MonotonicMixer(n_agents, state_dim, hidden_dim=cfg.mixer_hidden).to(device)
    tgt_mixer = MonotonicMixer(n_agents, state_dim, hidden_dim=cfg.mixer_hidden).to(device)
    tgt_mixer.load_state_dict(mixer.state_dict())

    opt = optim.Adam([p for q in q_nets for p in q.parameters()] + list(mixer.parameters()), lr=cfg.learning_rate)

    rb = MultiAgentReplay(cfg.buffer_size)

    logger = RunLogger(
        run_id=f"qmix_lite_{cfg.env}_s{seed}",
        chain="marl",
        algo="qmix_lite",
        env=cfg.env,
        seed=seed,
        budget={"steps": cfg.total_steps},
        out_dir=out_dir,
    )

    step_count = 0
    ep_reward = 0.0

    while step_count < cfg.total_steps:
        eps = linear_schedule(
            cfg.exploration_start,
            cfg.exploration_final,
            step_count,
            int(cfg.total_steps * cfg.exploration_fraction),
        )

        joint_actions = {}
        obs_stack = []
        for i, a in enumerate(agents):
            o = obs_dict[a]
            obs_stack.append(o)
            if np.random.rand() < eps:
                act = np.random.randint(act_dim)
            else:
                with torch.no_grad():
                    qv = q_nets[i](torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0))
                    act = int(qv.argmax(dim=1).item())
            joint_actions[a] = act

        next_obs, rewards, terms, truncs, infos = env.step(joint_actions)
        done = all(terms.values()) or all(truncs.values())
        reward = float(np.mean([rewards.get(a, 0.0) for a in agents]))
        ep_reward += reward

        rb.add(
            {
                "obs": np.array(obs_stack, dtype=np.float32),
                "state": np.concatenate(obs_stack, axis=0).astype(np.float32),
                "actions": np.array([joint_actions[a] for a in agents], dtype=np.int64),
                "reward": reward,
                "next_obs": np.array([next_obs.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agents], dtype=np.float32),
                "next_state": np.concatenate([next_obs.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agents], axis=0).astype(np.float32),
                "done": float(done),
            }
        )

        obs_dict = next_obs
        step_count += 1

        if step_count > cfg.learning_starts and len(rb) >= cfg.batch_size and step_count % cfg.train_freq == 0:
            batch = rb.sample(cfg.batch_size)
            obs_b = torch.tensor(np.stack([b["obs"] for b in batch]), dtype=torch.float32, device=device)
            state_b = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32, device=device)
            act_b = torch.tensor(np.stack([b["actions"] for b in batch]), dtype=torch.int64, device=device)
            rew_b = torch.tensor(np.array([b["reward"] for b in batch]), dtype=torch.float32, device=device).unsqueeze(1)
            nobs_b = torch.tensor(np.stack([b["next_obs"] for b in batch]), dtype=torch.float32, device=device)
            nstate_b = torch.tensor(np.stack([b["next_state"] for b in batch]), dtype=torch.float32, device=device)
            done_b = torch.tensor(np.array([b["done"] for b in batch]), dtype=torch.float32, device=device).unsqueeze(1)

            agent_qs = []
            next_agent_qs = []
            for i in range(n_agents):
                q_vals = q_nets[i](obs_b[:, i, :])
                q_taken = q_vals.gather(1, act_b[:, i].unsqueeze(1))
                agent_qs.append(q_taken)

                with torch.no_grad():
                    next_q_vals = tgt_q_nets[i](nobs_b[:, i, :])
                    next_q = next_q_vals.max(dim=1, keepdim=True)[0]
                    next_agent_qs.append(next_q)

            agent_qs_t = torch.cat(agent_qs, dim=1)
            next_agent_qs_t = torch.cat(next_agent_qs, dim=1)
            q_tot = mixer(agent_qs_t, state_b)
            with torch.no_grad():
                q_tot_next = tgt_mixer(next_agent_qs_t, nstate_b)
                target = rew_b + (1.0 - done_b) * cfg.gamma * q_tot_next

            loss = ((q_tot - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for q in q_nets for p in q.parameters()] + list(mixer.parameters()), 10.0)
            opt.step()

        if step_count % cfg.target_update_interval == 0:
            for q, tq in zip(q_nets, tgt_q_nets):
                tq.load_state_dict(q.state_dict())
            tgt_mixer.load_state_dict(mixer.state_dict())

        if done:
            logger.log_train(step_count, ep_reward)
            obs_dict, _ = env.reset()
            ep_reward = 0.0

    # Lightweight eval: use last train episodes summary
    eval_curve = [p["reward"] for p in logger.train_curve[-20:]]
    mean = float(np.mean(eval_curve)) if eval_curve else 0.0
    std = float(np.std(eval_curve)) if eval_curve else 0.0
    ci95 = float(1.96 * std / np.sqrt(max(len(eval_curve), 1))) if eval_curve else 0.0
    payload = logger.finalize({"mean_reward": mean, "std_reward": std, "ci95": ci95, "episodes": len(eval_curve)})

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_nets": [q.state_dict() for q in q_nets],
            "mixer": mixer.state_dict(),
        },
        out_dir / "model.pt",
    )
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
