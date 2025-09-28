"""
Environments Module

This module provides custom reinforcement learning environments
for various RL paradigms and experimental setups.
"""

from .environments import (
    BaseEnvironment,
    ContinuousMountainCar,
    PredatorPreyEnvironment,
    CausalBanditEnvironment,
    QuantumControlEnvironment,
    FederatedLearningEnvironment,
)

__all__ = [
    "BaseEnvironment",
    "ContinuousMountainCar",
    "PredatorPreyEnvironment",
    "CausalBanditEnvironment",
    "QuantumControlEnvironment",
    "FederatedLearningEnvironment",
]
