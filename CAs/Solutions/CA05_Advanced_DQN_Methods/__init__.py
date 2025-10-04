"""
CA5 Advanced DQN Methods - Complete Implementation

This package contains implementations of advanced Deep Q-Network (DQN) methods
including Double DQN, Dueling DQN, Prioritized Experience Replay, and Rainbow DQN.

Components:
- agents/: DQN agent implementations
- environments/: Custom environment definitions
- utils/: Utility functions and analysis tools
- experiments/: Experiment configurations and runners
- evaluation/: Performance evaluation utilities
- training_examples.py: Training examples and comparisons
- CA5.ipynb: Jupyter notebook with interactive examples
"""

__version__ = "1.0.0"
__author__ = "CA5 Advanced DQN Methods Team"

# Import main components
from .agents import (
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    PrioritizedDQNAgent,
    RainbowDQNAgent,
)

from .environments import (
    GridWorldEnv,
    MountainCarContinuousEnv,
    LunarLanderEnv,
    make_env,
)

from .utils import ReplayBuffer, PrioritizedReplayBuffer, QNetwork, DuelingQNetwork

# Main training function
from .training_examples import train_dqn_agent, dqn_variant_comparison

# Experiment and evaluation utilities
from .experiments import ExperimentRunner, ExperimentConfig, get_dqn_configs
from .evaluation import PerformanceEvaluator, compare_agents

__all__ = [
    # Agents
    "DQNAgent",
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "PrioritizedDQNAgent",
    "RainbowDQNAgent",
    # Environments
    "GridWorldEnv",
    "MountainCarContinuousEnv",
    "LunarLanderEnv",
    "make_env",
    # Utilities
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "QNetwork",
    "DuelingQNetwork",
    # Training
    "train_dqn_agent",
    "dqn_variant_comparison",
    # Experiments
    "ExperimentRunner",
    "ExperimentConfig",
    "get_dqn_configs",
    # Evaluation
    "PerformanceEvaluator",
    "compare_agents",
]


def get_version():
    """Get package version"""
    return __version__


def get_info():
    """Get package information"""
    return {
        "name": "CA5 Advanced DQN Methods",
        "version": __version__,
        "author": __author__,
        "description": "Complete implementation of advanced DQN methods",
        "components": [
            "Vanilla DQN",
            "Double DQN",
            "Dueling DQN",
            "Prioritized Experience Replay",
            "Rainbow DQN",
        ],
    }


# Quick start function
def quick_start():
    """Quick start example"""
    print("CA5 Advanced DQN Methods - Quick Start")
    print("=" * 40)
    print("1. Import agents: from agents import DQNAgent")
    print("2. Create environment: env = gym.make('CartPole-v1')")
    print("3. Initialize agent: agent = DQNAgent(state_dim, action_dim)")
    print("4. Train agent: train_dqn_agent('CartPole-v1', 'dqn')")
    print("5. Run experiments: python run.sh")
    print("=" * 40)


if __name__ == "__main__":
    quick_start()

