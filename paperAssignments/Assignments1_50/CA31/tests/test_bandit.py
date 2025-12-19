"""Unit tests for CA31 bandit example.

Focus:
- Determinism: same seed -> same results
- Basic learning: agent prefers the best arm after many steps
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ca31.utils import set_seed, load_config
from ca31.bandit import BernoulliBandit, EpsilonGreedyAgent
from ca31.train import run_experiment


def test_deterministic_run_same_seed():
    cfg = {"seed": 123, "arm_probs": [0.1, 0.9], "n_steps": 500, "epsilon": 0.0}
    rng1 = set_seed(cfg["seed"])
    r1 = run_experiment(cfg, rng1)

    rng2 = set_seed(cfg["seed"])
    r2 = run_experiment(cfg, rng2)

    assert r1["rewards"] == r2["rewards"]
    assert r1["actions"] == r2["actions"]
    assert r1["total_reward"] == r2["total_reward"]


def test_agent_learns_to_prefer_best_arm():
    cfg = {"seed": 0, "arm_probs": [0.1, 0.9], "n_steps": 1000, "epsilon": 0.05}
    rng = set_seed(cfg["seed"])
    r = run_experiment(cfg, rng)

    # after training, the best action estimate should be the true best arm (index 1)
    assert r["best_action_estimate"] == 1
    # average reward should be significantly larger than random uniform baseline (0.5*(0.1+0.9)=0.5)
    assert r["avg_reward"] > 0.6
