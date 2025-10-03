"""CA14 Advanced Deep Reinforcement Learning Package.

This package contains implementations of advanced RL algorithms including:
- Offline RL (Conservative Q-Learning, Implicit Q-Learning)
- Safe RL (Constrained Policy Optimization, Lagrangian methods)
- Multi-Agent RL (MADDPG, QMIX)
- Robust RL (Domain Randomization, Adversarial Training)
- Comprehensive evaluation framework
"""

from . import (
    offline_rl,
    safe_rl,
    multi_agent,
    robust_rl,
    environments,
    evaluation,
    utils,
)

__version__ = "1.0.0"
__author__ = "Advanced RL Research Group"

__all__ = [
    "offline_rl",
    "safe_rl",
    "multi_agent",
    "robust_rl",
    "environments",
    "evaluation",
    "utils",
]
