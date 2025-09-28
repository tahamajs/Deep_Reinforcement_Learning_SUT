"""
Reinforcement Learning GridWorld Implementation

This package contains a complete implementation of reinforcement learning
algorithms for the GridWorld environment, including:

- GridWorld environment (environments.py)
- Policy classes (policies.py)
- RL algorithms (algorithms.py)
- Visualization functions (visualization.py)
- Experiment functions (experiments.py)
"""

__version__ = "1.0.0"

from .environments import GridWorld, create_custom_environment
from .policies import (
    Policy,
    RandomPolicy,
    GreedyPolicy,
    CustomPolicy,
    GreedyActionPolicy,
    create_policy,
)
from .algorithms import (
    policy_evaluation,
    compute_q_from_v,
    compute_v_from_q,
    policy_improvement,
    policy_iteration,
    value_iteration,
    q_learning,
)
from .visualization import (
    plot_value_function,
    plot_policy,
    plot_q_values,
    plot_learning_curve,
    plot_value_iteration_convergence,
    compare_policies,
)
from .experiments import (
    experiment_discount_factors,
    experiment_policy_comparison,
    experiment_policy_iteration,
    experiment_value_iteration,
    experiment_q_learning,
    experiment_environment_modifications,
    run_all_experiments,
)

__all__ = [
    # Environments
    "GridWorld",
    "create_custom_environment",
    # Policies
    "Policy",
    "RandomPolicy",
    "GreedyPolicy",
    "CustomPolicy",
    "GreedyActionPolicy",
    "create_policy",
    # Algorithms
    "policy_evaluation",
    "compute_q_from_v",
    "compute_v_from_q",
    "policy_improvement",
    "policy_iteration",
    "value_iteration",
    "q_learning",
    # Visualization
    "plot_value_function",
    "plot_policy",
    "plot_q_values",
    "plot_learning_curve",
    "plot_value_iteration_convergence",
    "compare_policies",
    # Experiments
    "experiment_discount_factors",
    "experiment_policy_comparison",
    "experiment_policy_iteration",
    "experiment_value_iteration",
    "experiment_q_learning",
    "experiment_environment_modifications",
    "run_all_experiments",
]
