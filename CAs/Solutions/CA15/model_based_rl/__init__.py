"""
Model-Based Reinforcement Learning Module

Provides implementations of model-based RL algorithms including dynamics models,
uncertainty quantification, model predictive control, and Dyna-Q learning.
"""

from .algorithms import (
    DynamicsModel,
    ModelEnsemble,
    ModelPredictiveController,
    DynaQAgent,
)

__all__ = ["DynamicsModel", "ModelEnsemble", "ModelPredictiveController", "DynaQAgent"]
