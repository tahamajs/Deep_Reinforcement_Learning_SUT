"""
CA12: Multi-Agent Reinforcement Learning and Advanced Policy Methods

This package contains implementations of multi-agent RL algorithms including:
- Multi-Agent Actor-Critic (MAAC)
- Value Decomposition Networks (VDN)
- Counterfactual Multi-Agent Policy Gradients (COMA)
- Proximal Policy Optimization (PPO) variants
- Distributed RL and communication protocols

The package is organized into modular components for better maintainability.
"""

__version__ = "1.0.0"
__author__ = "DRL Course Team"
__description__ = "Multi-Agent RL and Advanced Policy Methods"

from .agents import (
    MultiAgentActor,
    MultiAgentCritic,
    VDNAgent,
    COMAAgent,
)

from .utils import (
    set_seed,
    get_device,
)

__all__ = [
    "MultiAgentActor",
    "MultiAgentCritic",
    "VDNAgent",
    "COMAAgent",
    "set_seed",
    "get_device",
]
