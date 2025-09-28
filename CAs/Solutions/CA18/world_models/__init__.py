"""
World Models Module

This module provides world model-based reinforcement learning algorithms
including recurrent state space models (RSSM), model predictive control (MPC),
and imagination-augmented agents (I2A).
"""

from .world_models import (
    RSSMCore,
    WorldModel,
    MPCPlanner,
    ImaginationAugmentedAgent
)

__all__ = [
    'RSSMCore',
    'WorldModel',
    'MPCPlanner',
    'ImaginationAugmentedAgent'
]