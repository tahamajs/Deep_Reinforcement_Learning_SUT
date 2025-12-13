import torch
from typing import Dict, Any

class Config:
    """
    Configuration class for Advanced Policy Gradient Methods (CA9).
    Centralizes all hyperparameters for models, agents, and training.
    """

    # General
    SEED: int = 42
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment
    ENV_NAME_DISCRETE: str = "CartPole-v1"
    ENV_NAME_CONTINUOUS: str = "Pendulum-v1"
    MAX_STEPS_DISCRETE: int = 500
    MAX_STEPS_CONTINUOUS: int = 200

    # Network Architecture
    HIDDEN_DIM: int = 128

    # REINFORCE Agent
    REINFORCE_GAMMA: float = 0.99
    REINFORCE_LR: float = 1e-3
    REINFORCE_NUM_EPISODES: int = 500

    # Actor-Critic Agent
    AC_GAMMA: float = 0.99
    AC_GAE_LAMBDA: float = 0.95
    AC_LR: float = 3e-4
    AC_VALUE_COEFF: float = 0.5
    AC_ENTROPY_COEFF: float = 0.01

    # PPO Agent
    PPO_GAMMA: float = 0.99
    PPO_GAE_LAMBDA: float = 0.95
    PPO_CLIP_RATIO: float = 0.2
    PPO_LR: float = 3e-4
    PPO_VALUE_COEFF: float = 0.5
    PPO_ENTROPY_COEFF: float = 0.01
    PPO_EPOCHS: int = 10
    PPO_BATCH_SIZE: int = 64
    PPO_UPDATE_FREQ: int = 2048 # Number of steps to collect before updating

    # Continuous PPO Agent
    CONTINUOUS_PPO_ACTION_BOUND: float = 1.0 # Will be overwritten by env.action_space.high[0]
    CONTINUOUS_PPO_NUM_EPISODES: int = 500
    CONTINUOUS_PPO_UPDATE_FREQ: int = 2048

    # Visualization and Analysis
    NUM_RUNS_COMPARISON: int = 3
    NUM_EPISODES_COMPARISON: int = 200
    SAVE_DIR: str = "visualizations/"

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Converts class attributes to a dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("__") and not callable(getattr(cls, key))
        }

# Instantiate config for easier access
config = Config()

