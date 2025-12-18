"""Training script for CA1: Annealed Implicit Sinkhorn (AIS-DRL) minimal integration.

This script implements the pseudocode from the CA1 README in a compact, runnable form.
- Supports both low-dim (MLP encoder) and pixel (simple conv encoder) observations.
- Uses an optional target network (periodic hard update) for stability.
- Uses AnnealedSinkhornLoss as the distributional loss between particle clouds.

Not intended to be a full research harness; it's a readable reference implementation you can run
for smoke tests and extend for large-scale experiments.

Usage example:
    python train.py --env CartPole-v1 --seed 0 --steps 20000 --batch 64

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


def to_tensor(x, device):
    return torch.from_numpy(x).to(device)


def select_action_from_particles(
    particles: torch.Tensor, epsilon: float = 0.0
) -> np.ndarray:
    # particles: (B, A, N, D) -> compute mean over particles and last dim -> (B, A)
    with torch.no_grad():
        means = particles.mean(dim=2).mean(dim=-1)  # (B, A)
        best = torch.argmax(means, dim=-1).cpu().numpy()
    # For single env (B=1) return int
    return best


def train(cfg: Config):
    os.makedirs(cfg.save_dir, exist_ok=True)

    env = gym.make(cfg.env)
    eval_env = gym.make(cfg.env)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env.reset(seed=cfg.seed)
    eval_env.reset(seed=cfg.seed + 1)

    obs_example, _ = env.reset()
    obs_example = preprocess_obs(obs_example)

    # Decide encoder type
    if obs_example.ndim == 3:
        # pixel obs (H,W,C) or (C,H,W) - convert to C,H,W
        # ensure channels-first
        if obs_example.shape[-1] in (1, 3):
            # H,W,C -> C,H,W
            obs_shape = (
                obs_example.shape[2],
                obs_example.shape[0],
                obs_example.shape[1],
            )
        else:
            # assume already C,H,W
            obs_shape = obs_example.shape
        pixel = True
    else:
        obs_shape = obs_example.shape
        pixel = False

    # Build networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For discrete action spaces
    if hasattr(env.action_space, "n"):
        num_actions = env.action_space.n
    else:
        raise NotImplementedError(
            "Only discrete action spaces supported in this minimal trainer"
        )

    if pixel:
        # use provided NatureCNN
        encoder = NatureCNN(in_channels=obs_shape[0], out_dim=512)
        model = ParticleQNetwork(
            in_channels=obs_shape[0],
            num_actions=num_actions,
            num_particles=cfg.num_particles,
            particle_dim=cfg.particle_dim,
        )
        target_model = copy.deepcopy(model)
    else:
        # low-dim: MLP encoder + particle head
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

    # Replay buffer expects flattened obs arrays matching preprocess
    replay = ReplayBuffer(obs_shape, size=cfg.replay_size)

    # warmup
    obs, _ = env.reset()
    obs = preprocess_obs(obs)
    if pixel:
        if obs.shape[-1] in (1, 3):
            obs = np.transpose(obs, (2, 0, 1))
    step = 0
    ep_ret = 0.0
    ep_len = 0

    print("Starting training on", cfg.env, "device=", device)
    while step < cfg.steps:
        # epsilon greedy (simple linear decay)
        epsilon = max(0.01, 0.4 - 0.4 * (step / cfg.steps))

        # select action
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            # compute particles for current obs
            obs_t = np.expand_dims(obs, 0).astype(np.float32)
            if pixel and obs_t.dtype == np.uint8:
                obs_t = obs_t.astype(np.float32) / 255.0
            if pixel and obs_t.shape[1] != obs_shape[0]:
                obs_t = np.transpose(obs_t, (0, 3, 1, 2))
            obs_tensor = torch.from_numpy(obs_t).to(device)
            particles = model(obs_tensor)  # (1, A, N, D)
            best = select_action_from_particles(particles)
            a = int(best[0])

        next_obs, r, done, truncated, info = (
            env.step(a) if "truncated" in env.step.__code__.co_varnames else env.step(a)
        )
        # gym API differences: some return (obs, reward, terminated, truncated, info)
        # normalize to (obs, rew, done)
        if isinstance(next_obs, tuple):
            # unexpected; safeguard
            next_obs = next_obs[0]
        # Older gym returns (obs, reward, done, info)
        if isinstance(done, tuple):
            # guard - shouldn't happen
            done = bool(done[0])

        next_obs = preprocess_obs(next_obs)
        if pixel and next_obs.shape[-1] in (1, 3):
            next_obs = np.transpose(next_obs, (2, 0, 1))

        replay.add(obs, a, r, next_obs, done)

        obs = next_obs
        ep_ret += r
        ep_len += 1
        step += 1

        if done:
            obs, _ = env.reset()
            obs = preprocess_obs(obs)
            if pixel and obs.shape[-1] in (1, 3):
                obs = np.transpose(obs, (2, 0, 1))
            print(f"Episode finished - len={ep_len} ret={ep_ret:.2f} step={step}")
            ep_ret = 0.0
            ep_len = 0

        # update
        if step > cfg.update_start and step % cfg.update_every == 0:
            for _ in range(1):
                o_batch, a_batch, r_batch, no_batch, d_batch = replay.sample(
                    cfg.batch_size
                )
                # to tensors
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

                # model outputs particles: (B, A, N, D)
                pred_particles = model(o_t)  # (B, A, N, D)

                # select predicted for actions taken
                B = pred_particles.shape[0]
                N = pred_particles.shape[2]
                D = pred_particles.shape[3]
                # gather
                idx = a_t.view(-1, 1, 1, 1).expand(-1, 1, N, D)
                pred_sel = pred_particles.gather(1, idx).squeeze(1)  # (B, N, D)

                # target: use target_model to compute next state's best action and its particles
                with torch.no_grad():
                    next_particles = target_model(no_t)
                    next_means = next_particles.mean(dim=2).mean(dim=-1)
                    next_best = torch.argmax(next_means, dim=-1)  # (B,)
                    idx_n = next_best.view(-1, 1, 1, 1).expand(-1, 1, N, D)
                    targ_sel = next_particles.gather(1, idx_n).squeeze(1)  # (B, N, D)
                    # bellman target y = r + gamma * z' * (1 - done)
                    y = r_t.view(-1, 1, 1) + cfg.gamma * targ_sel * (
                        1.0 - d_t.view(-1, 1, 1)
                    )

                # compute sinkhorn loss between pred_sel and y
                loss = loss_fn(pred_sel, y)

                opt.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                # step annealing in loss module
                loss_fn.step_annealing(1)

        # periodic target update
        if step % cfg.target_update_every == 0:
            target_model.load_state_dict(model.state_dict())

        # evaluation / checkpoint
        if step % cfg.eval_interval == 0:
            print(f"Step {step}: running evaluation")
            eval_returns = []
            for _ in range(3):
                ep_ret_eval = 0.0
                obs_e, _ = eval_env.reset()
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
                    so = eval_env.step(action)
                    if len(so) == 5:
                        obs_e, r_e, term, trunc, info = so
                        done_e = term or trunc
                    else:
                        obs_e, r_e, done_e, info = so
                    obs_e = preprocess_obs(obs_e)
                    if pixel and obs_e.shape[-1] in (1, 3):
                        obs_e = np.transpose(obs_e, (2, 0, 1))
                    ep_ret_eval += float(r_e)
                eval_returns.append(ep_ret_eval)
            mean_eval = float(np.mean(eval_returns))
            print(f"Eval returns (3 episodes): mean={mean_eval:.2f}")
            # save checkpoint
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
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--replay", type=int, default=50000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--particles", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    return Config(
        env=args.env,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch,
        replay_size=args.replay,
        warmup_steps=args.warmup,
        num_particles=args.particles,
        lr=args.lr,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

