"""
Agents Module

This module contains next-generation RL agent implementations
for advanced reinforcement learning paradigms.
"""

from .multi_agent_rl import (
    MultiAgentReplayBuffer,
    MADDPGActor,
    MADDPGCritic,
    MADDPGAgent,
    CommunicationNetwork,
    CommMADDPG,
    PredatorPreyEnvironment,
)

from .federated_rl import (
    DifferentialPrivacy,
    GradientCompression,
    FederatedRLClient,
    FederatedRLServer,
)

from .quantum_rl import (
    QuantumGate,
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    RotationX,
    RotationY,
    RotationZ,
    CNOT,
    QuantumCircuit,
    VariationalQuantumCircuit,
    QuantumStateEncoder,
    QuantumPolicy,
    QuantumValueNetwork,
    QuantumRLAgent,
)

from .advanced_safety import (
    SafetyMonitor,
    ConstrainedPolicyOptimization,
)

__all__ = [
    "MultiAgentReplayBuffer",
    "MADDPGActor",
    "MADDPGCritic",
    "MADDPGAgent",
    "CommunicationNetwork",
    "CommMADDPG",
    "PredatorPreyEnvironment",
    "DifferentialPrivacy",
    "GradientCompression",
    "FederatedRLClient",
    "FederatedRLServer",
    "QuantumGate",
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "RotationX",
    "RotationY",
    "RotationZ",
    "CNOT",
    "QuantumCircuit",
    "VariationalQuantumCircuit",
    "QuantumStateEncoder",
    "QuantumPolicy",
    "QuantumValueNetwork",
    "QuantumRLAgent",
    "SafetyMonitor",
    "ConstrainedPolicyOptimization",
]
