from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def mean_std_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci95)}


def evaluate_agent(agent, env, episodes: int = 5, deterministic: bool = True) -> Dict[str, float]:
    rewards: List[float] = []
    costs: List[float] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        while not done:
            action = agent.act(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_cost += float(info.get("cost", 0.0)) if isinstance(info, dict) else 0.0
            done = terminated or truncated
        rewards.append(ep_reward)
        costs.append(ep_cost)

    stats = mean_std_ci95(rewards)
    out = {
        "mean_reward": stats["mean"],
        "std_reward": stats["std"],
        "ci95": stats["ci95"],
        "episodes": episodes,
    }
    if any(c != 0 for c in costs):
        out["mean_cost"] = float(np.mean(costs))
        out["violation_rate"] = float(np.mean(np.array(costs) > 0))
    return out
