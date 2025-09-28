"""
Policy Gradient Methods and Neural Networks in RL
CA4: Modular Implementation

This package provides a complete implementation of policy gradient methods
for reinforcement learning, including REINFORCE, Actor-Critic, and neural
network function approximation.
"""

from .environments import EnvironmentWrapper, create_environment, get_environment_info
from .policies import (
    PolicyNetwork,
    ValueNetwork,
    create_policy_network,
    test_policy_network,
)
from .algorithms import REINFORCEAgent, ActorCriticAgent, create_agent
from .visualization import (
    PolicyVisualizer,
    PolicyGradientMathVisualizer,
    TrainingVisualizer,
    plot_learning_curves,
)
from .exploration import ExplorationScheduler, create_exploration_strategy
from .experiments import PolicyGradientExperiment, run_quick_test

__version__ = "1.0.0"
__author__ = "Deep RL Course"
__description__ = "Policy Gradient Methods for Reinforcement Learning"

__all__ = [
    "EnvironmentWrapper",
    "create_environment",
    "get_environment_info",
    "PolicyNetwork",
    "ValueNetwork",
    "create_policy_network",
    "test_policy_network",
    "REINFORCEAgent",
    "ActorCriticAgent",
    "create_agent",
    "PolicyVisualizer",
    "PolicyGradientMathVisualizer",
    "TrainingVisualizer",
    "plot_learning_curves",
    "ExplorationScheduler",
    "create_exploration_strategy",
    "PolicyGradientExperiment",
    "run_quick_test",
]
