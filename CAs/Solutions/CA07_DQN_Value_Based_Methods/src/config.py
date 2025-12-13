"""
Configuration for Deep Q-Networks (DQN) and Value-Based Methods.

This file centralizes all hyperparameters for various DQN agents,
including network architectures, training parameters, and environment settings.
"""

import torch

class DQNConfig:
    """
    Configuration class for all DQN variants.
    """
    # General
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment
    ENV_NAME = "CartPole-v1"
    MAX_EPISODE_STEPS = 500  # Max steps per episode during training
    EVAL_MAX_STEPS = 500     # Max steps per episode during evaluation

    # Agent Parameters
    HIDDEN_DIM = 128         # Hidden layer dimension for Q-networks
    LR = 1e-3                # Learning rate for the optimizer
    GAMMA = 0.99             # Discount factor
    BATCH_SIZE = 64          # Batch size for training
    REPLAY_BUFFER_SIZE = 100000 # Capacity of the replay buffer
    TARGET_UPDATE_FREQ = 10  # Frequency to update the target network (in training steps)

    # Epsilon-greedy exploration (for Vanilla, Double, Dueling DQN)
    EPSILON_START = 1.0      # Initial epsilon value
    EPSILON_END = 0.01       # Final epsilon value
    EPSILON_DECAY = 0.995    # Epsilon decay rate per episode

    # Noisy Network parameters (for NoisyDQN)
    NOISE_STD = 0.1          # Standard deviation of noise for NoisyLinear layers

    # Prioritized Experience Replay (PER) parameters (for Rainbow DQN)
    PER_ALPHA = 0.6          # Prioritization exponent (0=uniform, 1=greedy)
    PER_BETA_START = 0.4     # Initial importance sampling bias correction
    PER_BETA_FRAMES = 100000 # Number of frames over which to anneal beta
    PER_EPS = 1e-6           # Small epsilon to prevent zero priority

    # N-step Q-learning parameters (for Rainbow DQN)
    N_STEPS = 3              # Number of steps for multi-step returns

    # Distributional RL (C51) parameters (for Rainbow DQN)
    V_MIN = -10.0            # Minimum value for value distribution
    V_MAX = 10.0             # Maximum value for value distribution
    N_ATOMS = 51             # Number of atoms in the value distribution

    # Training Loop Parameters
    TRAIN_EPISODES = 500     # Number of episodes for training
    EVAL_EPISODES = 10       # Number of episodes for evaluation
    LOG_INTERVAL = 50        # Log training progress every N episodes

    # Paths
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"
    VISUALIZATIONS_DIR = "visualizations"
    PICTURES_DIR = "pictures" # For notebook-generated plots
