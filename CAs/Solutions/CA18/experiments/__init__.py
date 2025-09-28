"""
Experiments Module for CA18 - Advanced RL Paradigms

This module provides comprehensive experiment frameworks for evaluating and comparing
advanced reinforcement learning algorithms, including quantum RL, causal RL,
multi-agent systems, and federated learning approaches.
"""

from .experiments import (
    BaseExperiment,
    QuantumRLExperiment,
    CausalRLExperiment,
    MultiAgentRLExperiment,
    FederatedRLExperiment,
    ComparativeExperimentRunner,
)

__all__ = [
    "BaseExperiment",
    "QuantumRLExperiment",
    "CausalRLExperiment",
    "MultiAgentRLExperiment",
    "FederatedRLExperiment",
    "ComparativeExperimentRunner",
]