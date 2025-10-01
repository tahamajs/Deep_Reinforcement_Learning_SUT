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

from .models import (
    VariationalAutoencoder,
    LatentDynamicsModel,
    RewardModel,
    WorldModel,
    RSSM,
)

from .agents import (
    LatentActor,
    LatentCritic,
    DreamerAgent,
)

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
