"""
CA07: Deep Q-Networks (DQN) and Value-Based Methods
===================================================

This package provides a comprehensive set of implementations for Deep Q-Networks (DQN)
and various advanced value-based reinforcement learning methods. It includes modularized
components for agents, neural networks, replay buffers, and utility functions,
facilitating structured development and analysis of DQN algorithms.
"""

__version__ = "1.0.0"
__author__ = "Deep Reinforcement Learning Course"

# Import main components from src subpackages
from .src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DuelingDoubleDQNAgent, NoisyDQNAgent, RainbowDQNAgent
from .src.models import QNetwork, DuelingQNetwork, NoisyLinear, NoisyQNetwork, CategoricalQNetwork
from .src.data import ReplayBuffer, PrioritizedReplayBuffer
from .src.utils import set_seed, smooth_curve
from .src.config import DQNConfig
from .src.losses import c51_loss

# You can also import specific training examples or evaluation tools if desired
from .training_examples import train_dqn_agent, compare_dqn_variants, plot_dqn_comparison, hyperparameter_optimization_study, robustness_analysis, advanced_dqn_training_demo

__all__ = [
    # Configuration
    "DQNConfig",
    # Agents
    "DQNAgent",
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "DuelingDoubleDQNAgent",
    "NoisyDQNAgent",
    "RainbowDQNAgent", # Add RainbowDQNAgent
    # Networks
    "QNetwork",
    "DuelingQNetwork",
    "NoisyLinear",
    "NoisyQNetwork",
    "CategoricalQNetwork", # Add CategoricalQNetwork
    # Data Structures
    "ReplayBuffer",
    "PrioritizedReplayBuffer", # Add PrioritizedReplayBuffer
    # Utilities
    "set_seed",
    "smooth_curve",
    # Loss functions
    "c51_loss", # Add c51_loss
    # Training and Analysis Functions
    "train_dqn_agent",
    "compare_dqn_variants",
    "plot_dqn_comparison",
    "hyperparameter_optimization_study",
    "robustness_analysis",
    "advanced_dqn_training_demo",
]


