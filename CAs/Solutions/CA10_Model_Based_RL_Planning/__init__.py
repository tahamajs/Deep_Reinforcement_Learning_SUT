"""
CA10: Model-Based Reinforcement Learning and Planning Methods
===========================================================

This package provides comprehensive implementations of Model-Based Reinforcement Learning
and Planning Methods including:

- Classical planning algorithms (Value/Policy Iteration)
- Dyna-Q algorithm with integrated planning and learning
- Monte Carlo Tree Search (MCTS)
- Model Predictive Control (MPC)
- Environment models (tabular and neural)
- Comprehensive evaluation and comparison frameworks

Author: DRL Course Team
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "DRL Course Team"

# Import main components
from .agents import *
from .environments import *
from .models import *
from .experiments import *
from .evaluation import *
from .utils import *

__all__ = [
    # Agents
    "DynaQAgent",
    "DynaQPlusAgent",
    "MCTSAgent",
    "MPCAgent",
    "ModelBasedPlanner",
    # Environments
    "SimpleGridWorld",
    "BlockingMaze",
    # Models
    "TabularModel",
    "NeuralModel",
    "ModelTrainer",
    # Experiments
    "ModelBasedComparisonFramework",
    # Evaluation
    "ModelBasedEvaluator",
    "PerformanceMetrics",
    # Utils
    "set_seed",
    "create_directories",
    "save_results",
    "load_results",
    "plot_learning_curves",
    "plot_comparison",
    "create_summary_plots",
]
