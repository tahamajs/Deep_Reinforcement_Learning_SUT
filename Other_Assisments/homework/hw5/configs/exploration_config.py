"""
Exploration Configuration

This file contains default hyperparameters and settings for exploration training.
"""

# Environment settings
ENV_NAME = "Pendulum-v0"

# Exploration hyperparameters
BONUS_COEFF = 1.0  # Exploration bonus coefficient

# Data collection
INITIAL_ROLLOUTS = 10
MAX_EPISODE_LENGTH = 1000

# Training settings
LOG_INTERVAL = 10
