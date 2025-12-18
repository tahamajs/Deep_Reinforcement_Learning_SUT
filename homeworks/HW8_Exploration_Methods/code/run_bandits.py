#!/usr/bin/env python3
"""
Run multi-armed bandit experiments (ε-greedy, UCB1, Thompson) and save comparison plots.

Usage:
    python run_bandits.py --num_arms 10 --steps 10000 --runs 100 --out ../pictures/bandit_comparison.png
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, means: np.ndarray) -> None:
        assert means.ndim == 1
        self.means = means
        self.num_arms = int(means.shape[0])
        self.optimal_mean = float(means.max())
        self.optimal_arm = int(np.argmax(means))

    @classmethod
    def random(
        cls, num_arms: int, seed: int = 0, low: float = 0.05, high: float = 0.95
    ):
        rng = np.random.default_rng(seed)
        means = rng.uniform(low=low, high=high, size=num_arms)
        return cls(means)

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self.means[arm])


class BasePolicy:
    def select(self, t: int) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        raise NotImplementedError


class EpsilonGreedy(BasePolicy):
    def __init__(self, num_arms: int, epsilon: float = 0.1, seed: int = 0) -> None:
        self.num_arms = num_arms
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(num_arms, dtype=int)
        self.values = np.zeros(num_arms, dtype=float)

    def select(self, t: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB1(BasePolicy):
    def __init__(self, num_arms: int, c: float = np.sqrt(2.0), seed: int = 0) -> None:
        self.num_arms = num_arms
        self.c = float(c)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(num_arms, dtype=int)
        self.values = np.zeros(num_arms, dtype=float)

    def select(self, t: int) -> int:
        # Pull each arm once initially
        for a in range(self.num_arms):
            if self.counts[a] == 0:
                return a
        confidence = self.c * np.sqrt(np.log(max(t, 1)) / (self.counts + 1e-12))
        return int(np.argmax(self.values + confidence))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonBeta(BasePolicy):
    def __init__(
        self, num_arms: int, alpha0: float = 1.0, beta0: float = 1.0, seed: int = 0
    ) -> None:
        self.num_arms = num_arms
        self.rng = np.random.default_rng(seed)
        self.alpha = np.full(num_arms, alpha0, dtype=float)
        self.beta = np.full(num_arms, beta0, dtype=float)

    def select(self, t: int) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        # reward expected to be 0/1
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward


PolicyResult = Tuple[np.ndarray, np.ndarray]  # (rewards, cumulative_regret)


def regret_curve(rewards: np.ndarray, optimal_mean: float) -> np.ndarray:
    inst_regret = optimal_mean - rewards
    return np.cumsum(inst_regret)


def run_policy(
    env: BernoulliBandit, policy: BasePolicy, horizon: int, seed: int
) -> PolicyResult:
    rng = np.random.default_rng(seed)
    rewards = np.zeros(horizon, dtype=float)
    for t in range(horizon):
        arm = policy.select(t + 1)
        r = env.pull(arm, rng)
        policy.update(arm, r)
        rewards[t] = r
    return rewards, regret_curve(rewards, env.optimal_mean)


def run_bandit_experiment(
    num_arms: int, steps: int, runs: int, seed: int = 0
) -> Dict[str, Dict]:
    results = {}
    for run in range(runs):
        env = BernoulliBandit.random(
            num_arms=num_arms, seed=seed + run, low=0.05, high=0.95
        )
        policies = {
            "epsilon_greedy": EpsilonGreedy(
                num_arms=num_arms, epsilon=0.1, seed=seed + run + 1
            ),
            "ucb1": UCB1(num_arms=num_arms, c=np.sqrt(2.0), seed=seed + run + 2),
            "thompson_beta": ThompsonBeta(
                num_arms=num_arms, alpha0=1.0, beta0=1.0, seed=seed + run + 3
            ),
        }
        for name, policy in policies.items():
            rewards, regrets = run_policy(env, policy, steps, seed + run)
            if run == 0:
                results[name] = {
                    "rewards": rewards.astype(float),
                    "regrets": regrets.astype(float),
                }
            else:
                results[name]["rewards"] += rewards
                results[name]["regrets"] += regrets

    # Average over runs
    for name in results:
        results[name]["rewards"] /= runs
        results[name]["regrets"] /= runs
    return results


def plot_bandit_comparison(results: Dict[str, Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        ax.plot(data["regrets"], label=name)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Exploration strategies on Bernoulli bandit")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run bandit experiments")
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="../pictures/bandit_comparison.png")
    args = parser.parse_args()

    results = run_bandit_experiment(
        args.num_arms, args.steps, args.runs, seed=args.seed
    )
    plot_bandit_comparison(results, args.out)


if __name__ == "__main__":
    main()



