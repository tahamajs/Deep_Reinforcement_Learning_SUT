"""Minimal training script demonstrating usage of `src` utilities.

This script is intentionally small and pedagogical: it runs a few on-policy
episodes, computes simple returns, fits a value function, and performs policy
gradient updates. It's import-safe and uses `argparse` for basic configuration.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import torch
import torch.optim as optim

from src.config import ExperimentConfig
from src.utils import set_seed, get_device, ensure_dir
from src.model import PolicyNetwork, ValueNetwork
from src.data import collect_episodes, discounts
from src.losses import policy_gradient_loss, value_loss, entropy_loss


def compute_returns(episodes: List, gamma: float) -> List[List[float]]:
    returns = []
    for ep in episodes:
        rewards = [t.reward for t in ep]
        R = discounts(rewards, gamma)
        returns.append(R)
    return returns


def run(config: ExperimentConfig):
    set_seed(config.seed)
    device = get_device(config.device)

    # Lazy import of gym to allow tests to run without env installed
    try:
        import gym
    except Exception:
        raise RuntimeError("`gym` is required to run `train.py`. Install gym.")

    env = gym.make(config.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes=config.hidden_sizes).to(device)
    value = ValueNetwork(obs_dim, hidden_sizes=config.hidden_sizes).to(device)

    opt = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=config.learning_rate)

    out_dir = ensure_dir(Path("results"))
    csv_path = out_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "episode_return"])

    for ep in range(1, config.max_episodes + 1):
        episodes = collect_episodes(config.env_name, policy, num_episodes=1)
        ep_data = episodes[0]
        ep_return = sum(t.reward for t in ep_data)
        R = compute_returns([ep_data], config.gamma)[0]

        obs = torch.tensor([t.obs for t in ep_data], dtype=torch.float32, device=device)
        actions = torch.tensor([t.action for t in ep_data], dtype=torch.long, device=device)
        returns = torch.tensor(R, dtype=torch.float32, device=device)

        # Compute value predictions
        values = value(obs)
        advantages = returns - values.detach()

        # Compute log probs
        logits = policy(obs)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        logp = dist.log_prob(actions)

        pgloss = policy_gradient_loss(logp, advantages)
        vloss = value_loss(values, returns)
        # optional entropy regularization
        ent = entropy_loss(logits)
        loss = pgloss + 0.5 * vloss + config.entropy_coef * ent

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Simple logging
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep, ep_return])

        if ep % max(1, config.max_episodes // 10) == 0:
            print(f"Episode {ep}/{config.max_episodes}: return={ep_return:.2f}")

    # Save a small checkpoint
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    torch.save({"policy_state": policy.state_dict(), "value_state": value.state_dict()}, ckpt_dir / "last.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()
    run(cfg)


if __name__ == "__main__":
    main()
