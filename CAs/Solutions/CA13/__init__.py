"""
CA13: Sample-Efficient Deep RL

This package contains implementations of sample-efficient RL algorithms including:
- Model-free agents (DQN, PPO)
- Model-based agents with learned dynamics
- Sample-efficient methods (Rainbow, SAC)
- Hierarchical RL approaches
- Advanced replay buffers

The package is organized into modular components for better maintainability.
"""

__version__ = "1.0.0"
__author__ = "DRL Course Team"
__description__ = "Sample-Efficient Deep Reinforcement Learning"

from .agents import (
    ModelFreeAgent,
    DQNAgent,
    ModelBasedAgent,
)

from .buffers import ReplayBuffer

from .utils import (
    set_seed,
    get_device,
)

__all__ = [
    "ModelFreeAgent",
    "DQNAgent",
    "ModelBasedAgent",
    "ReplayBuffer",
    "set_seed",
    "get_device",
]
