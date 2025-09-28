"""
CA15: Advanced Deep Reinforcement Learning

This package contains implementations of advanced deep RL algorithms including:
- Model-based RL (Dynamics models, MPC, Dyna-Q)
- Hierarchical RL (Options, Goal-conditioned learning, Feudal networks)
- Advanced Planning (MCTS, Model-based value expansion, World models)
- Experimental frameworks and utilities

The package is organized into modular components for better maintainability and reusability.
"""

__version__ = "1.0.0"
__author__ = "Advanced RL Research"
__description__ = "Advanced Deep Reinforcement Learning Algorithms and Experiments"

# Import main classes for easy access
from .model_based_rl.algorithms import (
    DynamicsModel,
    ModelEnsemble,
    ModelPredictiveController,
    DynaQAgent,
)

from .hierarchical_rl.algorithms import (
    Option,
    HierarchicalActorCritic,
    GoalConditionedAgent,
    FeudalNetwork,
)

from .hierarchical_rl.environments import HierarchicalRLEnvironment

from .planning.algorithms import (
    MCTSNode,
    MonteCarloTreeSearch,
    ModelBasedValueExpansion,
    LatentSpacePlanner,
    WorldModel,
)

from .environments.grid_world import SimpleGridWorld

from .experiments.runner import ExperimentRunner
from .experiments.hierarchical import HierarchicalRLExperiment
from .experiments.planning import PlanningAlgorithmsExperiment

from .utils import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    RunningStats,
    Logger,
    NeuralNetworkUtils,
    VisualizationUtils,
    EnvironmentUtils,
    ExperimentUtils,
    set_device,
    get_device,
    to_tensor,
)


# Package-level utilities
def get_version():
    """Get package version."""
    return __version__


def list_algorithms():
    """List all available algorithms in the package."""
    algorithms = {
        "Model-Based RL": [
            "DynamicsModel",
            "ModelEnsemble",
            "ModelPredictiveController",
            "DynaQAgent",
        ],
        "Hierarchical RL": [
            "Option",
            "HierarchicalActorCritic",
            "GoalConditionedAgent",
            "FeudalNetwork",
        ],
        "Planning": [
            "MonteCarloTreeSearch",
            "ModelBasedValueExpansion",
            "LatentSpacePlanner",
            "WorldModel",
        ],
    }
    return algorithms


def create_experiment(name: str, **kwargs):
    """Factory function to create experiments."""
    experiments = {
        "hierarchical": HierarchicalRLExperiment,
        "planning": PlanningAlgorithmsExperiment,
        "general": ExperimentRunner,
    }

    if name not in experiments:
        raise ValueError(
            f"Unknown experiment: {name}. Available: {list(experiments.keys())}"
        )

    return experiments[name](**kwargs)


# Setup device on import
set_device()

__all__ = [
    # Model-based RL
    "DynamicsModel",
    "ModelEnsemble",
    "ModelPredictiveController",
    "DynaQAgent",
    # Hierarchical RL
    "Option",
    "HierarchicalActorCritic",
    "GoalConditionedAgent",
    "FeudalNetwork",
    "HierarchicalRLEnvironment",
    # Planning
    "MCTSNode",
    "MonteCarloTreeSearch",
    "ModelBasedValueExpansion",
    "LatentSpacePlanner",
    "WorldModel",
    # Environments
    "SimpleGridWorld",
    # Experiments
    "ExperimentRunner",
    "HierarchicalRLExperiment",
    "PlanningAlgorithmsExperiment",
    # Utilities
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "RunningStats",
    "Logger",
    "NeuralNetworkUtils",
    "VisualizationUtils",
    "EnvironmentUtils",
    "ExperimentUtils",
    "set_device",
    "get_device",
    "to_tensor",
    # Package functions
    "get_version",
    "list_algorithms",
    "create_experiment",
]
