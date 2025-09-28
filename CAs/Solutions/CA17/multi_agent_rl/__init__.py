"""
Multi-Agent Reinforcement Learning Module

This module contains implementations of multi-agent reinforcement learning algorithms,
including MADDPG, communication networks, and multi-agent environments.
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

__all__ = [
    "MultiAgentReplayBuffer",
    "MADDPGActor",
    "MADDPGCritic",
    "MADDPGAgent",
    "CommunicationNetwork",
    "CommMADDPG",
    "PredatorPreyEnvironment",
]
