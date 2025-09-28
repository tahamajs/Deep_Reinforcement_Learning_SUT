"""
DDPG Implementation Package

This package contains modular components for DDPG, TD3, and HER algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

from .actor import ActorNetwork
from .critic import CriticNetwork, CriticNetworkTD3
from .replay_buffer import ReplayBuffer
from .ddpg_agent import DDPGAgent, EpsilonNormalActionNoise

__all__ = [
    "ActorNetwork",
    "CriticNetwork",
    "CriticNetworkTD3",
    "ReplayBuffer",
    "DDPGAgent",
    "EpsilonNormalActionNoise",
]
