"""
CA07: Deep Q-Networks (DQN) and Value-Based Methods
===================================================
"""

__version__ = "1.0.0"
__author__ = "Deep Reinforcement Learning Course"

# Import main classes
from .agents.core import DQNAgent, QNetwork, ReplayBuffer
from .agents.double_dqn import DoubleDQNAgent
from .agents.dueling_dqn import DuelingDQNAgent, DuelingQNetwork, DuelingDoubleDQNAgent
from .agents.utils import DQNVisualizer, DQNAnalyzer

# Import utilities
from .utils import (
    set_seed,
    smooth_curve,
    calculate_statistics,
    find_convergence_episode,
    PerformanceTracker,
    ExperimentLogger,
    create_summary_plot,
    benchmark_agent,
    save_results,
    load_results,
    print_experiment_summary,
)

# Import environments
from .environments import (
    RewardShapingWrapper,
    StateNormalizationWrapper,
    ActionRepeatWrapper,
    EpisodeStatisticsWrapper,
    create_cartpole_env,
    create_mountain_car_env,
    create_acrobot_env,
    get_environment_info,
    test_environment,
)

# Import evaluation tools
from .evaluation import DQNEvaluator, evaluate_training_progress

# Import models
from .models import (
    QNetwork,
    DuelingQNetwork,
    NoisyLinear,
    NoisyQNetwork,
    CategoricalQNetwork,
    RainbowQNetwork,
    create_model,
    count_parameters,
)

__all__ = [
    # Core agents
    "DQNAgent",
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "DuelingDoubleDQNAgent",
    # Networks
    "QNetwork",
    "DuelingQNetwork",
    "NoisyQNetwork",
    "CategoricalQNetwork",
    "RainbowQNetwork",
    # Utilities
    "ReplayBuffer",
    "DQNVisualizer",
    "DQNAnalyzer",
    "DQNEvaluator",
    # Environment wrappers
    "RewardShapingWrapper",
    "StateNormalizationWrapper",
    "ActionRepeatWrapper",
    "EpisodeStatisticsWrapper",
    # Environment creators
    "create_cartpole_env",
    "create_mountain_car_env",
    "create_acrobot_env",
    # Utility functions
    "set_seed",
    "smooth_curve",
    "calculate_statistics",
    "find_convergence_episode",
    "PerformanceTracker",
    "ExperimentLogger",
    "create_summary_plot",
    "benchmark_agent",
    "save_results",
    "load_results",
    "print_experiment_summary",
    # Evaluation
    "evaluate_training_progress",
    # Model utilities
    "create_model",
    "count_parameters",
    "get_model_info",
]


