"""
Environments Module

This module provides custom reinforcement learning environments
for various RL paradigms and experimental setups.
"""

from .environments import (
    # Base Classes
    BaseEnvironment,
    # Continuous Control
    ContinuousMountainCar,
    # Multi-Agent
    PredatorPreyEnvironment,
    # Causal RL
    CausalBanditEnvironment,
    # Quantum RL
    QuantumControlEnvironment,
    # Federated Learning
    FederatedLearningEnvironment,
)

__all__ = [
    # Base Classes
    "BaseEnvironment",
    # Continuous Control
    "ContinuousMountainCar",
    # Multi-Agent
    "PredatorPreyEnvironment",
    # Causal RL
    "CausalBanditEnvironment",
    # Quantum RL
    "QuantumControlEnvironment",
    # Federated Learning
    "FederatedLearningEnvironment",
]
