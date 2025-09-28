"""
SAC (Soft Actor-Critic) Configuration

This file contains default hyperparameters and settings for SAC training.
"""

# Environment settings
ENV_NAME = "Pendulum-v0"

# Network architecture
HIDDEN_SIZES = [256, 256]

# Training hyperparameters
LEARNING_RATE = 3e-3
ALPHA = 1.0  # Temperature parameter
BATCH_SIZE = 256
DISCOUNT = 0.99
TAU = 0.01  # Soft update coefficient

# Training settings
TOTAL_STEPS = 100000
MAX_EPISODE_LENGTH = 1000
LOG_INTERVAL = 10

# SAC specific
REPARAMETERIZE = True