"""
Training script for MaxSink (CA8) with minibatch updates, TensorBoard and optional W&B.
Supports vectorized envs via gymnasium.vector.SyncVectorEnv.
"""

import argparse
import time
from collections import deque
from typing import Deque, Dict, Any, Callable

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

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


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

    try:
        import gymnasium as gym
        from gymnasium.vector import SyncVectorEnv
    except Exception:
        print("Could not import gymnasium; please install required env packages.")
        return

    def make_env_fn(env_id: str, beta: float) -> Callable:
        def _fn():
            e = gym.make(env_id)
            return ToTheMaxWrapper(e, beta=beta)

        return _fn

    # create vectorized envs if requested
    if cfg.use_vector_env and cfg.num_envs > 1:
        env_fns = [make_env_fn(cfg.env_name, cfg.beta) for _ in range(cfg.num_envs)]
        env = SyncVectorEnv(env_fns)
        is_vector = True
    else:
        env = ToTheMaxWrapper(gym.make(cfg.env_name), beta=cfg.beta)
        is_vector = False

    reset_res = env.reset()
    if is_vector:
        obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        state_dim = np.asarray(obs[0]).ravel().shape[0]
    else:
        obs, info = reset_res
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

    # logging
    writer = SummaryWriter(cfg.tb_logdir)
    if cfg.use_wandb and wandb is not None:
        wandb.init(project="MaxSink-CA8", config=cfg.as_dict())

    total_steps = args.steps
    start_time = time.time()

    # If vectorized, maintain per-env states
    if is_vector:
        states = np.asarray(obs)
        num_envs = states.shape[0]
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lens = np.zeros(num_envs, dtype=np.int32)
    else:
        state = np.asarray(obs).ravel()
        episode_return = 0.0
        episode_len = 0

    for step in range(total_steps):
        # act (vectorized or single)
        if is_vector:
            actions = []
            for s in states:
                a = agent.act(torch.from_numpy(np.asarray(s).ravel()).float())
                actions.append(a)
            actions = np.asarray(actions)
            res = env.step(actions)
            if len(res) == 5:
                next_obs, rewards, terminated, truncated, infos = res
                dones = np.logical_or(terminated, truncated)
            else:
                next_obs, rewards, dones, infos = res
            for i in range(len(next_obs)):
                r = rewards[i]
                info = infos[i] if isinstance(infos, (list, tuple)) else infos
                next_state = np.asarray(next_obs[i]).ravel()
                episode_returns[i] += float(r)
                episode_lens[i] += 1
                r_store = info.get("reward_max", r) if isinstance(info, dict) else r
                replay.add(
                    {
                        "state": np.asarray(states[i]).ravel().astype(np.float32),
                        "action": np.array(actions[i], dtype=np.int64),
                        "reward": np.array([r_store], dtype=np.float32),
                        "next_state": next_state.astype(np.float32),
                        "done": np.array(float(dones[i]), dtype=np.float32),
                    }
                )
            states = np.asarray(next_obs)
        else:
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

        # updates: perform gradient steps with minibatches
        if (
            step > cfg.start_updates
            and step % cfg.update_every == 0
            and len(replay.buffer) >= cfg.batch_size
        ):
            # sample a big batch and optionally split into minibatches
            big_batch = replay.sample(cfg.batch_size)
            # split indices
            B = cfg.batch_size
            mb = cfg.minibatch_size
            num_mbs = max(1, B // mb)
            losses = []
            for _ in range(cfg.num_grad_steps):
                for i in range(num_mbs):
                    # create sub-batch
                    start = i * mb
                    end = start + mb
                    sub_batch = {
                        "state": big_batch["state"][start:end],
                        "action": big_batch["action"][start:end],
                        "reward": big_batch["reward"][start:end],
                        "next_state": big_batch["next_state"][start:end],
                        "done": big_batch["done"][start:end],
                    }
                    loss, sinkhorn_mean = agent.update(sub_batch)
                    losses.append(loss)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            # logging
            writer.add_scalar("train/sinkhorn_loss", mean_loss, step)
            if cfg.use_wandb and wandb is not None:
                wandb.log({"train/sinkhorn_loss": mean_loss, "step": step})
            if step % 1000 == 0:
                print(f"step={step} mean_loss={mean_loss:.4f}")

    writer.close()
    if cfg.use_wandb and wandb is not None:
        wandb.finish()
    print("Training loop finished. Time:", time.time() - start_time)


if __name__ == "__main__":
    main()

