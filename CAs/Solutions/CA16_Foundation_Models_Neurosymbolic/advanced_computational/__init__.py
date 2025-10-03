"""
Advanced Computational Paradigms for Deep Reinforcement Learning

This module contains implementations of advanced computational approaches including:
- Quantum-inspired RL algorithms
- Neuromorphic computing architectures
- Distributed and federated RL
- Energy-efficient learning
- Hybrid computing paradigms
"""

from .quantum_rl import (
    QuantumInspiredRL,
    QuantumStateRepresentation,
    QuantumAmplitudeEstimation,
)

from .neuromorphic import NeuromorphicNetwork, SpikingNeuralNetwork, STDPLearning

from .federated_rl import FederatedRLAggregator, FederatedClient, FederatedServer

from .energy_efficient import EnergyEfficientRL, AdaptiveComputation, EarlyExitNetwork

from .hybrid_computing import (
    HybridComputingAgent,
    QuantumClassicalHybrid,
    NeuromorphicClassicalHybrid,
)

__all__ = [
    "QuantumInspiredRL",
    "QuantumStateRepresentation",
    "QuantumAmplitudeEstimation",
    "NeuromorphicNetwork",
    "SpikingNeuralNetwork",
    "STDPLearning",
    "FederatedRLAggregator",
    "FederatedClient",
    "FederatedServer",
    "EnergyEfficientRL",
    "AdaptiveComputation",
    "EarlyExitNetwork",
    "HybridComputingAgent",
    "QuantumClassicalHybrid",
    "NeuromorphicClassicalHybrid",
]
