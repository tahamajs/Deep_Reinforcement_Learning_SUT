"""Training script for CA1: Annealed Implicit Sinkhorn (AIS-DRL) minimal integration.

This script implements the pseudocode from the CA1 README in a compact, runnable form.
- Supports both low-dim (MLP encoder) and pixel (simple conv encoder) observations.
- Uses an optional target network (periodic hard update) for stability.
- Uses AnnealedSinkhornLoss as the distributional loss between particle clouds.

Not intended to be a full research harness; it's a readable reference implementation you can run
for smoke tests and extend for large-scale experiments.

Usage examples:
    python train.py --config paperAssignments/Assignments1-50/CA1/config.yaml
    python train.py --env CartPole-v1 --steps 20000 --batch 64

"""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from paperAssignments.Assignments1_50.CA1.sinkhorn import AnnealedSinkhornLoss
from paperAssignments.Assignments1_50.CA1.model import (
    ParticleHead,
    ParticleQNetwork,
    NatureCNN,
)
from paperAssignments.Assignments1_50.CA1.wrappers import make_atari_env
from paperAssignments.Assignments1_50.CA1.env_utils import reset_env, step_env


# optional YAML loader
try:
    import yaml
except Exception:
    yaml = None


@dataclass
class Config:
    env: str = "CartPole-v1"
    seed: int = 0
    steps: int = 100_000
    batch_size: int = 256
    replay_size: int = 50_000
    warmup_steps: int = 1_000
    update_every: int = 1
    update_start: int = 1_000
    gamma: float = 0.99
    lr: float = 1e-4
    num_particles: int = 64
    particle_dim: int = 1
    sinkhorn_iters: int = 20
    sinkhorn_eps_start: float = 1.0
    sinkhorn_eps_end: float = 0.01
    sinkhorn_decay_steps: int = 100_000
    target_update_every: int = 1000
    eval_interval: int = 5000
    save_dir: str = "runs/ca1"


class ReplayBuffer:
    def __init__(self, obs_shape, size: int = 100000):
        self.size = size
        self.obs_shape = obs_shape
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.acts = np.zeros((size,), dtype=np.int64)
        self.rews = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.next_obs[self.ptr] = no
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.done[self.ptr] = float(d)
        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def sample(self, batch_size: int):
        max_idx = self.size if self.full else self.ptr
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (
            self.obs[idxs],
            self.acts[idxs],
            self.rews[idxs],
            self.next_obs[idxs],
            self.done[idxs],
        )


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def preprocess_obs(o):
    # Convert env obs to float32 numpy array and normalize images if present
    o = np.array(o)
    if o.dtype == np.uint8:
        o = o.astype(np.float32) / 255.0
    return o


def select_action_from_particles(
    particles: torch.Tensor, epsilon: float = 0.0
) -> np.ndarray:
    # particles: (B, A, N, D) -> compute mean over particles and last dim -> (B, A)
    with torch.no_grad():
        means = particles.mean(dim=2).mean(dim=-1)  # (B, A)
        best = torch.argmax(means, dim=-1).cpu().numpy()
    return best


def train(cfg: Config):
    os.makedirs(cfg.save_dir, exist_ok=True)

    # For Atari-like env ids use wrappers; detect by name heuristic
    use_atari = isinstance(cfg.env, str) and (
        "NoFrameskip" in cfg.env
        or "Atari" in cfg.env
        or cfg.env.lower() in ["pong", "breakout", "seaquest"]
    )

    if use_atari:
        env = make_atari_env(cfg.env)
        eval_env = make_atari_env(cfg.env)
    else:
        env = gym.make(cfg.env)
        eval_env = gym.make(cfg.env)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # use reset_env to support gym/gymnasium
    try:
        reset_env(env, seed=cfg.seed)
        reset_env(eval_env, seed=cfg.seed + 1)
    except Exception:
        pass

    obs_example = reset_env(env)
    obs_example = preprocess_obs(obs_example)

    # Decide encoder type
    if obs_example.ndim == 3:
        # pixel obs (H,W,C) or (C,H,W) - convert to C,H,W
        if obs_example.shape[-1] in (1, 3):
            obs_shape = (
                obs_example.shape[2],
                obs_example.shape[0],
                obs_example.shape[1],
            )
        else:
            obs_shape = obs_example.shape
        pixel = True
    else:
        obs_shape = obs_example.shape
        pixel = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(env.action_space, "n"):
        num_actions = env.action_space.n
    else:
        raise NotImplementedError(
            "Only discrete action spaces supported in this minimal trainer"
        )

    if pixel:
        model = ParticleQNetwork(
            in_channels=obs_shape[0],
            num_actions=num_actions,
            num_particles=cfg.num_particles,
            particle_dim=cfg.particle_dim,
        )
        target_model = copy.deepcopy(model)
    else:
        obs_dim = obs_shape[0]
        enc = MLPEncoder(obs_dim, out_dim=512)
        head = ParticleHead(
            in_dim=512,
            num_actions=num_actions,
            num_particles=cfg.num_particles,
            particle_dim=cfg.particle_dim,
        )

        class LowDimParticleNet(nn.Module):
            def __init__(self, enc, head):
                super().__init__()
                self.enc = enc
                self.head = head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                feats = self.enc(x)
                return self.head(feats)

        model = LowDimParticleNet(enc, head)
        target_model = copy.deepcopy(model)

    model.to(device)
    target_model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    loss_fn = AnnealedSinkhornLoss(
        n_iters=cfg.sinkhorn_iters,
        eps_start=cfg.sinkhorn_eps_start,
        eps_end=cfg.sinkhorn_eps_end,
        decay_steps=cfg.sinkhorn_decay_steps,
    )
    loss_fn.to(device)

    replay = ReplayBuffer(obs_shape, size=cfg.replay_size)

    obs = reset_env(env)
    obs = preprocess_obs(obs)
    if pixel and obs.shape[-1] in (1, 3):
        obs = np.transpose(obs, (2, 0, 1))

    step = 0
    ep_ret = 0.0
    ep_len = 0

    print("Starting training on", cfg.env, "device=", device)
    while step < cfg.steps:
        epsilon = max(0.01, 0.4 - 0.4 * (step / cfg.steps))

        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            obs_t = np.expand_dims(obs, 0).astype(np.float32)
            if pixel and obs_t.dtype == np.uint8:
                obs_t = obs_t.astype(np.float32) / 255.0
            if pixel and obs_t.shape[1] != obs_shape[0]:
                obs_t = np.transpose(obs_t, (0, 3, 1, 2))
            obs_tensor = torch.from_numpy(obs_t).to(device)
            particles = model(obs_tensor)
            best = select_action_from_particles(particles)
            a = int(best[0])

        next_obs, r, done, info = step_env(env, a)

        next_obs = preprocess_obs(next_obs)
        if pixel and next_obs.shape[-1] in (1, 3):
            next_obs = np.transpose(next_obs, (2, 0, 1))

        replay.add(obs, a, r, next_obs, done)

        obs = next_obs
        ep_ret += r
        ep_len += 1
        step += 1

        if done:
            obs = reset_env(env)
            obs = preprocess_obs(obs)
            if pixel and obs.shape[-1] in (1, 3):
                obs = np.transpose(obs, (2, 0, 1))
            print(f"Episode finished - len={ep_len} ret={ep_ret:.2f} step={step}")
            ep_ret = 0.0
            ep_len = 0

        if step > cfg.update_start and step % cfg.update_every == 0:
            o_batch, a_batch, r_batch, no_batch, d_batch = replay.sample(cfg.batch_size)
            o_batch = o_batch.astype(np.float32)
            no_batch = no_batch.astype(np.float32)

            if pixel and o_batch.ndim == 4 and o_batch.shape[1] != obs_shape[0]:
                o_batch = np.transpose(o_batch, (0, 3, 1, 2))
                no_batch = np.transpose(no_batch, (0, 3, 1, 2))

            o_t = torch.from_numpy(o_batch).to(device)
            no_t = torch.from_numpy(no_batch).to(device)
            a_t = torch.from_numpy(a_batch).to(device)
            r_t = torch.from_numpy(r_batch).to(device)
            d_t = torch.from_numpy(d_batch).to(device)

            pred_particles = model(o_t)

            B = pred_particles.shape[0]
            N = pred_particles.shape[2]
            D = pred_particles.shape[3]
            idx = a_t.view(-1, 1, 1, 1).expand(-1, 1, N, D)
            pred_sel = pred_particles.gather(1, idx).squeeze(1)

            with torch.no_grad():
                next_particles = target_model(no_t)
                next_means = next_particles.mean(dim=2).mean(dim=-1)
                next_best = torch.argmax(next_means, dim=-1)
                idx_n = next_best.view(-1, 1, 1, 1).expand(-1, 1, N, D)
                targ_sel = next_particles.gather(1, idx_n).squeeze(1)
                y = r_t.view(-1, 1, 1) + cfg.gamma * targ_sel * (
                    1.0 - d_t.view(-1, 1, 1)
                )

            loss = loss_fn(pred_sel, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            loss_fn.step_annealing(1)

        if step % cfg.target_update_every == 0:
            target_model.load_state_dict(model.state_dict())

        if step % cfg.eval_interval == 0:
            print(f"Step {step}: running evaluation")
            eval_returns = []
            for _ in range(3):
                ep_ret_eval = 0.0
                obs_e = reset_env(eval_env)
                obs_e = preprocess_obs(obs_e)
                if pixel and obs_e.shape[-1] in (1, 3):
                    obs_e = np.transpose(obs_e, (2, 0, 1))
                done_e = False
                while not done_e:
                    obs_t = np.expand_dims(obs_e, 0).astype(np.float32)
                    if pixel and obs_t.shape[1] != obs_shape[0]:
                        obs_t = np.transpose(obs_t, (0, 3, 1, 2))
                    obs_tensor = torch.from_numpy(obs_t).to(device)
                    particles = model(obs_tensor)
                    action = int(select_action_from_particles(particles)[0])
                    obs_e, r_e, done_e, info = step_env(eval_env, action)
                    obs_e = preprocess_obs(obs_e)
                    if pixel and obs_e.shape[-1] in (1, 3):
                        obs_e = np.transpose(obs_e, (2, 0, 1))
                    ep_ret_eval += float(r_e)
                eval_returns.append(ep_ret_eval)
            mean_eval = float(np.mean(eval_returns))
            print(f"Eval returns (3 episodes): mean={mean_eval:.2f}")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "step": step,
                },
                os.path.join(cfg.save_dir, f"ckpt_step_{step}.pt"),
            )

    env.close()
    eval_env.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--env", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--replay", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    cfg = Config()
    if args.config is not None:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load config files. Install with `pip install pyyaml`."
            )
        with open(args.config, "r") as f:
            data = yaml.safe_load(f)
        # shallow map
        for k, v in data.items():
            if k == "sinkhorn" and isinstance(v, dict):
                cfg.sinkhorn_iters = v.get("iters", cfg.sinkhorn_iters)
                cfg.sinkhorn_eps_start = v.get("eps_start", cfg.sinkhorn_eps_start)
                cfg.sinkhorn_eps_end = v.get("eps_end", cfg.sinkhorn_eps_end)
                cfg.sinkhorn_decay_steps = v.get(
                    "decay_steps", cfg.sinkhorn_decay_steps
                )
            elif hasattr(cfg, k):
                setattr(cfg, k, v)

    # CLI overrides
    if args.env is not None:
        cfg.env = args.env
    if args.seed is not None:
        cfg.seed = args.seed
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.replay is not None:
        cfg.replay_size = args.replay
    if args.warmup is not None:
        cfg.warmup_steps = args.warmup
    if args.particles is not None:
        cfg.num_particles = args.particles
    if args.lr is not None:
        cfg.lr = args.lr

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)















