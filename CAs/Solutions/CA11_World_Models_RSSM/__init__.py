"""
CA11: Advanced Model-Based RL and World Models

This package contains implementations of advanced model-based RL including:
- Variational Autoencoders for world models
- Recurrent State Space Models (RSSM)
- Dreamer agent architecture
- Latent space planning and imagination
- World model training and evaluation

The package is organized into modular components for better maintainability.
"""

__version__ = "1.0.0"
__author__ = "DRL Course Team"
__description__ = "Advanced Model-Based RL and World Models"

from .models.vae import VariationalAutoencoder
from .models.dynamics import LatentDynamicsModel
from .models.reward_model import RewardModel
from .models.world_model import WorldModel
from .models.rssm import RSSM

from .agents.latent_actor import LatentActor
from .agents.latent_critic import LatentCritic
from .agents.dreamer_agent import DreamerAgent

from .utils import (
    collect_world_model_data,
    set_seed,
    get_device,
)

__all__ = [
    "VariationalAutoencoder",
    "LatentDynamicsModel",
    "RewardModel",
    "WorldModel",
    "RSSM",
    "LatentActor",
    "LatentCritic",
    "DreamerAgent",
    "collect_world_model_data",
    "set_seed",
    "get_device",
]
