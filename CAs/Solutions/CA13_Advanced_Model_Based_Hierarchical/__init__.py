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
    HybridDynaAgent,
    SampleEfficientAgent,
    DataAugmentationDQN,
    OptionsCriticAgent,
    FeudalAgent,
)

from .buffers import ReplayBuffer, PrioritizedReplayBuffer

from .utils import (
    set_seed,
    get_device,
)

# Import training utilities
from .training_examples import (
    train_dqn_agent,
    train_model_based_agent,
    evaluate_agent,
    env_reset,
    env_step,
    EpisodeMetrics,
)

# Import evaluation framework
from .evaluation.advanced_evaluator import (
    AdvancedRLEvaluator,
    IntegratedAdvancedAgent,
)

# Import visualization utilities
from .utils import visualization

def get_version():
    """Get package version."""
    return __version__

__all__ = [
    "ModelFreeAgent",
    "DQNAgent",
    "ModelBasedAgent",
    "HybridDynaAgent",
    "SampleEfficientAgent",
    "DataAugmentationDQN",
    "OptionsCriticAgent",
    "FeudalAgent",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "set_seed",
    "get_device",
    "train_dqn_agent",
    "train_model_based_agent",
    "evaluate_agent",
    "env_reset",
    "env_step",
    "EpisodeMetrics",
    "AdvancedRLEvaluator",
    "IntegratedAdvancedAgent",
    "visualization",
    "get_version",
]
