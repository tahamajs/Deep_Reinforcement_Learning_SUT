"""
Configuration for Temporal Difference Learning Algorithms.

This file centralizes all hyperparameters for the GridWorld environment,
TD(0), Q-Learning, SARSA agents, and exploration strategies.
"""

import numpy as np

class GridWorldConfig:
    """Configuration for the GridWorld environment."""
    SIZE = 4
    START_STATE = (0, 0)
    GOAL_STATE = (3, 3)
    OBSTACLES = [(1, 1), (1, 2), (2, 1)]
    STEP_REWARD = -1
    GOAL_REWARD = 10
    OBSTACLE_REWARD = -5
    # For visualization
    VALUES_TITLE = "GridWorld Environment Layout"
    POLICY_TITLE = "Optimal Policy"

class AgentConfig:
    """Base configuration for TD learning agents."""
    ALPHA = 0.1         # Learning rate
    GAMMA = 0.9         # Discount factor
    NUM_EPISODES = 1000 # Number of episodes for training
    PRINT_EVERY = 100   # How often to print progress during training

class ExplorationConfig:
    """Configuration for exploration strategies."""
    EPSILON_START = 1.0 # Initial epsilon for epsilon-greedy
    EPSILON_MIN = 0.01  # Minimum epsilon
    EPSILON_DECAY = 0.995 # Decay rate for epsilon
    TEMPERATURE_START = 2.0 # Initial temperature for Boltzmann exploration
    TEMPERATURE_MIN = 0.1 # Minimum temperature
    TEMPERATURE_DECAY = 0.995 # Decay rate for temperature

class ExperimentConfig:
    """Configuration for running experiments."""
    NUM_RUNS = 5        # Number of independent runs for statistical analysis
    EVAL_EPISODES = 100 # Number of episodes for policy evaluation

class VisualizationConfig:
    """Configuration for visualizations."""
    FIGURE_SIZE = (12, 8)
    FONT_SIZE = 11
    PLOT_DPI = 300
    SAVE_DIR = "visualizations" # Directory to save plots
    
# Global Seed for reproducibility
SEED = 42
np.random.seed(SEED)


