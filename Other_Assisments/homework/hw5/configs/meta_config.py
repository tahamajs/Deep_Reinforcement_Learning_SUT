"""
Meta-Learning Configuration

This file contains default hyperparameters and settings for meta-learning training.
"""

# Environment settings
ENV_NAME = "Pendulum-v0"

# Network architecture
HIDDEN_SIZES = [256, 256]

# Training hyperparameters
LEARNING_RATE = 3e-3
META_LEARNING_RATE = 1e-3
DISCOUNT = 0.99

# Meta-learning specific
ADAPTATION_STEPS = 5
META_BATCH_SIZE = 4
NUM_TASKS = 20
META_STEPS = 100

# Training settings
MAX_EPISODE_LENGTH = 1000
LOG_INTERVAL = 10

# Algorithm selection
ALGORITHM = "maml"  # "maml" or "meta"