from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from grad_rl.core import RunLogger, mean_std_ci95, set_seed
from grad_rl.envs import make_env


@dataclass
class DynaQConfig:
    env: str = "CliffWalking-v0"
    episodes: int = 2000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    planning_steps: int = 20


class TabularAgent:
    def __init__(self, n_actions: int):
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        self.n_actions = n_actions

    def act(self, state, epsilon: float = 0.1):
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q[state]))


class EvalPolicy:
    def __init__(self, q_table):
        self.q = q_table

    def act(self, obs, deterministic: bool = True):
        return int(np.argmax(self.q[obs]))


def train_dyna_q(config: Dict, out_dir: Path, seed: int) -> Dict:
    cfg = DynaQConfig(**{k: v for k, v in config.items() if k in DynaQConfig.__annotations__})
    set_seed(seed)
    env = make_env(cfg.env, seed=seed)
    agent = TabularAgent(env.action_space.n)
    model = defaultdict(lambda: defaultdict(list))

    logger = RunLogger(
        run_id=f"dyna_q_{cfg.env.replace('/', '_')}_s{seed}",
        chain="model_based",
        algo="dyna_q",
        env=cfg.env,
        seed=seed,
        budget={"episodes": cfg.episodes},
        out_dir=out_dir,
    )

    rewards = []
    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs, epsilon=cfg.epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)

            agent.q[obs][action] += cfg.alpha * (
                reward + cfg.gamma * np.max(agent.q[next_obs]) - agent.q[obs][action]
            )
            model[obs][action].append((reward, next_obs))

            for _ in range(cfg.planning_steps):
                s = np.random.choice(list(model.keys()))
                a = np.random.choice(list(model[s].keys()))
                r, ns = model[s][a][np.random.randint(len(model[s][a]))]
                agent.q[s][a] += cfg.alpha * (r + cfg.gamma * np.max(agent.q[ns]) - agent.q[s][a])

            obs = next_obs

        rewards.append(total)
        logger.log_train(ep, total)

    eval_policy = EvalPolicy(agent.q)
    eval_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    eval_stats = mean_std_ci95(eval_rewards)
    payload = logger.finalize(
        {
            "mean_reward": eval_stats["mean"],
            "std_reward": eval_stats["std"],
            "ci95": eval_stats["ci95"],
            "episodes": len(eval_rewards),
        },
        extra={"train_summary": {"avg_last_100": float(np.mean(eval_rewards)) if eval_rewards else 0.0}},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    serializable_q = {str(k): v.tolist() for k, v in agent.q.items()}
    with (out_dir / "q_table.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_q, f, indent=2)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return payload
