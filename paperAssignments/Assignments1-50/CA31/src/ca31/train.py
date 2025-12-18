"""Guarded training script for a reproducible bandit experiment."""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Any, Optional

import numpy as np

from .utils import load_config, set_seed
from .bandit import BernoulliBandit, EpsilonGreedyAgent


def run_experiment(cfg: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Run a bandit experiment according to `cfg` and return results.

    cfg keys (defaults):
      - seed: int
      - arm_probs: list[float]
      - n_steps: int
      - epsilon: float
      - save_path: optional path to save CSV of rewards
    """
    if rng is None:
        rng = set_seed(cfg.get("seed", None))

    bandit = BernoulliBandit(cfg["arm_probs"])  # will raise on bad config
    agent = EpsilonGreedyAgent(n_arms=bandit.n_arms, epsilon=cfg.get("epsilon", 0.1))

    n_steps = int(cfg.get("n_steps", 1000))
    rewards = []
    actions = []

    for t in range(n_steps):
        a = agent.select_action(rng)
        r = bandit.pull(a, rng)
        agent.update(a, r)
        rewards.append(r)
        actions.append(a)

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    results = {
        "avg_reward": avg_reward,
        "total_reward": int(sum(rewards)),
        "rewards": rewards,
        "actions": actions,
        "final_values": agent.values,
        "best_action_estimate": agent.best_action(),
    }

    save_path = cfg.get("save_path")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # save rewards to CSV for reproducibility
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "action", "reward"])
            for i, (a, r) in enumerate(zip(actions, rewards)):
                writer.writerow([i, a, r])

    return results


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a CA31 bandit experiment")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rng = set_seed(cfg.get("seed", None))
    results = run_experiment(cfg, rng)
    print(f"Average reward: {results['avg_reward']:.4f}, total: {results['total_reward']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
