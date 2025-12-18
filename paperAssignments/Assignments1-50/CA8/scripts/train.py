"""
Training script skeleton for MaxSink (CA8).
This is a runnable script but kept lightweight for quick sanity runs.
Do NOT execute automatically on import.
"""

import argparse
import time
from collections import deque
from typing import Deque, Dict, Any

import numpy as np
import torch
import os
import sys

# Make the local src/ directory importable when running this script directly.
THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import cfg  # type: ignore
from envs import ToTheMaxWrapper  # type: ignore
from agent import MaxSinkAgent  # type: ignore
from utils import set_seed  # type: ignore


class ReplayBuffer:
    """Minimal replay buffer storing numpy arrays."""

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer: Deque = deque(maxlen=capacity)

    def add(self, transition: Dict[str, Any]):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        # stack arrays
        def stack(key):
            return torch.from_numpy(np.stack([b[key] for b in batch])).float()

        state = stack("state")
        action = torch.from_numpy(np.stack([b["action"] for b in batch])).long()
        reward = stack("reward").squeeze(-1)
        next_state = stack("next_state")
        done = torch.from_numpy(np.stack([b["done"] for b in batch])).float()
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--steps", type=int, default=cfg.total_steps)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(cfg.device)

    # env creation is left to user; this is skeleton code
    try:
        import gymnasium as gym

        env = gym.make(cfg.env_name)
        env = ToTheMaxWrapper(env, beta=cfg.beta)
    except Exception:
        print("Could not create env automatically. Please instantiate env manually.")
        return

    obs, info = env.reset()
    state_dim = np.asarray(obs).ravel().shape[0]
    n_actions = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )

    agent = MaxSinkAgent(
        state_dim=state_dim, n_actions=int(n_actions), device=str(device)
    )
    replay = ReplayBuffer()

    total_steps = args.steps
    start_time = time.time()
    state = np.asarray(obs).ravel()
    episode_return = 0.0
    episode_len = 0

    for step in range(total_steps):
        # act
        s_tensor = torch.from_numpy(state).float()
        a = agent.act(s_tensor)
        res = env.step(a)
        if len(res) == 5:
            next_obs, r, terminated, truncated, info = res
            done = terminated or truncated
        else:
            next_obs, r, done, info = res

        next_state = np.asarray(next_obs).ravel()
        episode_return += float(r)
        episode_len += 1

        # store transformed reward (info['reward_max'] if present)
        r_store = info.get("reward_max", r)
        replay.add(
            {
                "state": state.astype(np.float32),
                "action": np.array(a, dtype=np.int64),
                "reward": np.array([r_store], dtype=np.float32),
                "next_state": next_state.astype(np.float32),
                "done": np.array(float(done), dtype=np.float32),
            }
        )

        state = next_state
        if done:
            obs, info = env.reset()
            state = np.asarray(obs).ravel()
            episode_return = 0.0
            episode_len = 0

        # updates
        if (
            step > cfg.start_updates
            and step % cfg.update_every == 0
            and len(replay.buffer) >= cfg.batch_size
        ):
            batch = replay.sample(cfg.batch_size)
            loss, sinkhorn_mean = agent.update(batch)
            # minimal logging
            if step % 1000 == 0:
                print(f"step={step} loss={loss:.4f} sinkhorn={sinkhorn_mean:.4f}")

    print("Training loop finished. Time:", time.time() - start_time)


if __name__ == "__main__":
    main()
