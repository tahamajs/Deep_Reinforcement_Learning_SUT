"""
CA8: Causal Reasoning and Multi-Modal Reinforcement Learning - Configuration
=============================================================================

This module defines all hyperparameters and configuration settings for the CA8 project.
Centralizing these parameters ensures consistency and simplifies experimentation.

Author: DRL Course Team
"""

import torch
from typing import Dict, List, Any, Tuple

# General
SEED: int = 42
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES: int = 500
MAX_STEPS_PER_EPISODE: int = 500
SAVE_DIR: str = "visualizations/"
RESULTS_DIR: str = "results/"
LOGS_DIR: str = "logs/"

# Agent Parameters
HIDDEN_DIM: int = 128
GAMMA: float = 0.99
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 64
UPDATE_FREQ: int = 10

# Environment Parameters (e.g., CartPole-v1)
ENV_NAME: str = "CartPole-v1"
STATE_DIM: int = 4 # Example for CartPole
ACTION_DIM: int = 2 # Example for CartPole

# Multi-Modal Specifics
MODAL_DIMS: Dict[str, int] = {
    "state": STATE_DIM,
    "visual": 64,  # Mock visual features dimension
    "textual": 32, # Mock textual features dimension
    "audio": 48 # Mock audio features dimension
}
FUSION_TYPE: str = "cross_attention" # Options: "early", "late", "cross_attention"
NUM_ATTENTION_HEADS: int = 4

# Causal Discovery Specifics
CAUSAL_ALGORITHM: str = "PC" # Options: "PC", "GES", "NOTEARS", "CAM", "LiNGAM"
PC_ALPHA: float = 0.05
GES_MAX_ITER: int = 100

# Curriculum Learning Specifics
CURRICULUM_STAGES: List[Dict[str, Any]] = [
    {
        "modalities": ["state"],
        "noise_level": 0.0,
        "causal_complexity": "simple",
    },
    {
        "modalities": ["state", "visual"],
        "noise_level": 0.1,
        "causal_complexity": "simple",
    },
    {
        "modalities": ["state", "visual", "textual"],
        "noise_level": 0.2,
        "causal_complexity": "simple",
    },
    {
        "modalities": ["state", "visual", "textual"],
        "noise_level": 0.3,
        "causal_complexity": "complex",
    },
]
EPISODES_PER_STAGE: int = 100

# Visualization parameters
PLOT_DPI: int = 300
FIGURE_SIZE_LARGE: Tuple[int, int] = (20, 12)
FIGURE_SIZE_MEDIUM: Tuple[int, int] = (16, 12)
FIGURE_SIZE_SMALL: Tuple[int, int] = (12, 10)


