"""
CA12: Multi-Agent Reinforcement Learning and Advanced Policy Methods

This package contains comprehensive implementations of multi-agent RL algorithms including:
- Multi-Agent Actor-Critic (MAAC) and MADDPG
- Value Decomposition Networks (VDN)
- Counterfactual Multi-Agent Policy Gradients (COMA)
- Proximal Policy Optimization (PPO) variants
- Soft Actor-Critic (SAC) and Trust Region Policy Optimization (TRPO)
- Distributed RL (A3C, IMPALA, Parameter Server)
- Communication and coordination mechanisms
- Meta-learning and adaptation (MAML, opponent modeling)
- Real-world applications and training frameworks

The package is organized into modular components for better maintainability and extensibility.
"""

__version__ = "1.0.0"
__author__ = "DRL Course Team"
__description__ = "Multi-Agent RL and Advanced Policy Methods"

# Import core utilities
from .utils.setup import (
    device,
    n_gpus,
    agent_colors,
    performance_colors,
    ma_config,
    policy_config,
    MultiAgentConfig,
    PolicyConfig,
)

# Import cooperative learning algorithms
from .agents.cooperative_learning import (
    Actor,
    Critic,
    MADDPGAgent,
    MADDPG,
    ReplayBuffer,
    VDNAgent,
    VDN,
)

# Import advanced policy methods
from .agents.advanced_policy import PPONetwork, PPOAgent, SACAgent, GAEBuffer

# Import distributed RL components
from .agents.distributed_rl import (
    ParameterServer,
    A3CWorker,
    IMPALALearner,
    DistributedPPOCoordinator,
    EvolutionaryStrategy,
)

# Import meta-learning components
from .agents.meta_learning import (
    MAMLAgent,
    OpponentModel,
    PopulationBasedTraining,
    SelfPlayTraining,
)

# Import experimental frameworks
from .experiments.game_theory import GameTheoryUtils, MultiAgentEnvironment

from .experiments.communication import (
    CommunicationChannel,
    AttentionCommunication,
    CoordinationMechanism,
    MarketBasedCoordination,
    HierarchicalCoordination,
    EmergentCommunicationAgent,
)

from .experiments.applications import (
    ResourceAllocationEnvironment,
    AutonomousVehicleEnvironment,
    SmartGridEnvironment,
    MultiAgentGameTheoryAnalyzer,
)

from .experiments.training_framework import MultiAgentTrainingOrchestrator

__all__ = [
    # Core utilities
    "device",
    "n_gpus",
    "agent_colors",
    "performance_colors",
    "ma_config",
    "policy_config",
    "MultiAgentConfig",
    "PolicyConfig",
    # Cooperative learning
    "Actor",
    "Critic",
    "MADDPGAgent",
    "MADDPG",
    "ReplayBuffer",
    "VDNAgent",
    "VDN",
    # Advanced policy methods
    "PPONetwork",
    "PPOAgent",
    "SACAgent",
    "GAEBuffer",
    # Distributed RL
    "ParameterServer",
    "A3CWorker",
    "IMPALALearner",
    "DistributedPPOCoordinator",
    "EvolutionaryStrategy",
    # Meta-learning
    "MAMLAgent",
    "OpponentModel",
    "PopulationBasedTraining",
    "SelfPlayTraining",
    # Experimental frameworks
    "GameTheoryUtils",
    "MultiAgentEnvironment",
    "CommunicationChannel",
    "AttentionCommunication",
    "CoordinationMechanism",
    "MarketBasedCoordination",
    "HierarchicalCoordination",
    "EmergentCommunicationAgent",
    "ResourceAllocationEnvironment",
    "AutonomousVehicleEnvironment",
    "SmartGridEnvironment",
    "MultiAgentGameTheoryAnalyzer",
    "MultiAgentTrainingOrchestrator",
]
