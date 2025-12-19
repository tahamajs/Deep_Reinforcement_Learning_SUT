"""Small deterministic A2C smoke-run for generating example figures.

This script runs a short A2C training on CartPole-v1 using the local
`A2CAgent` implementation in `src/agent.py` and saves per-step rewards to
`results/ca31_a2c_smoke_rewards.csv` and a small JSON summary. It is designed
for quick, deterministic runs suitable for producing demonstration figures.

Usage:
    python scripts/run_a2c_smoke.py --config configs/a2c.yaml --steps 2000

Notes:
- This is a small smoke run (few thousand steps) intended for creating quick
  illustrative plots for the report. It is NOT intended to reproduce full
  experimental results.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch

from src.agent import A2CAgent
from src.utils import set_seed
from src.config import load_config


def run_smoke(cfg: Dict[str, Any], out_dir: Path, total_steps: int = 2000) -> Dict[str, Any]:
    rng = set_seed(cfg.get("seed", None))

    env_name = cfg.get("env_name", "CartPole-v1")
    env = gym.make(env_name)

    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    agent = A2CAgent(num_inputs=obs_dim, num_actions=n_actions, config=cfg, device="cpu")

    step = 0
    episode_rewards = []
    episode_reward = 0.0
    episode_lengths = []

    # CSV export
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ca31_a2c_smoke_rewards.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "action", "reward"])  # header

        state, _ = env.reset(seed=int(cfg.get("seed", 0)))
        state = np.asarray(state, dtype=np.float32)

        # we collect small rollouts and call agent.update periodically
        rollout_states = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []

        while step < total_steps:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # (1, obs_dim)
            action, log_prob, value = agent.act(state_tensor)

            # convert single-element tensors to scalars / ints
            action_item = int(action.item())
            log_prob = log_prob.squeeze(0)

            next_state, reward, terminated, truncated, info = env.step(action_item)
            done = bool(terminated or truncated)

            writer.writerow([step, action_item, float(reward)])
            rollout_states.append(state_tensor.squeeze(0))
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value.squeeze(0))
            rollout_rewards.append(float(reward))
            rollout_dones.append(done)

            episode_reward += float(reward)

            state = np.asarray(next_state, dtype=np.float32)

            step += 1

            # update every cfg['num_steps'] or when done
            if len(rollout_rewards) >= int(cfg.get("num_steps", 5)) or done:
                rollouts = {
                    "states": rollout_states,
                    "actions": rollout_actions,
                    "log_probs": rollout_log_probs,
                    "values": rollout_values,
                    "rewards": rollout_rewards,
                    "dones": rollout_dones,
                }
                _ = agent.update(rollouts)

                rollout_states = []
                rollout_actions = []
                rollout_log_probs = []
                rollout_values = []
                rollout_rewards = []
                rollout_dones = []

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(info.get("episode", {}).get("l", 0) or len(episode_lengths))
                episode_reward = 0.0
                state, _ = env.reset()
                state = np.asarray(state, dtype=np.float32)

    # finalize
    env.close()

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    summary = {
        "avg_reward": avg_reward,
        "total_steps": total_steps,
        "episodes": len(episode_rewards),
        "final_best_action_estimate": agent,  # to inspect if needed
    }

    # Save JSON summary
    json_path = out_dir / "ca31_a2c_smoke_summary.json"
    with open(json_path, "w") as jf:
        json.dump({"avg_reward": avg_reward, "episodes": len(episode_rewards)}, jf, indent=2)

    return {"csv_path": str(csv_path), "summary_path": str(json_path)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/a2c.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="results/ca31_smoke/")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed

    out = run_smoke(cfg, Path(args.out_dir), total_steps=args.steps)
    print("Wrote:", out)


if __name__ == '__main__':
    main()
