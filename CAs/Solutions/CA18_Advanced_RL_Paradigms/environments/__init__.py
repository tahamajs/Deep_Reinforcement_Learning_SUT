"""
Environments Module for CA18 - Advanced RL Paradigms

This module provides advanced reinforcement learning environments that demonstrate
the various paradigms covered in CA18, including quantum control, causal bandits,
multi-agent quantum systems, and federated learning scenarios.
"""

from .environments import (
    QuantumEnvironment,
    CausalBanditEnvironment,
    MultiAgentQuantumEnvironment,
    FederatedLearningEnvironment,
)

__all__ = [
    "QuantumEnvironment",
    "CausalBanditEnvironment",
    "MultiAgentQuantumEnvironment",
    "FederatedLearningEnvironment",
]
