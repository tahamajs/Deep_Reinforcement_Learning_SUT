"""
SAC (Soft Actor-Critic) Configuration

This file contains default hyperparameters and settings for SAC training.
"""
ENV_NAME = "Pendulum-v0"
HIDDEN_SIZES = [256, 256]
LEARNING_RATE = 3e-3
ALPHA = 1.0
BATCH_SIZE = 256
DISCOUNT = 0.99
TAU = 0.01
TOTAL_STEPS = 100000
MAX_EPISODE_LENGTH = 1000
LOG_INTERVAL = 10
REPARAMETERIZE = True