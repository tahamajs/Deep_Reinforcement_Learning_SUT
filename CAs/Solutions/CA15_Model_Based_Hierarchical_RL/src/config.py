import os
import torch
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional

@dataclass
class GeneralConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    total_env_steps: int = 1_000_000
    batch_size: int = 256
    num_random_seeds: int = 5
    log_interval: int = 1000  # Log every N environment steps
    eval_interval: int = 10000 # Evaluate every N environment steps
    save_interval: int = 50000 # Save model every N environment steps
    results_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    pictures_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pictures")

@dataclass
class DynamicsModelConfig:
    state_dim: int = field(default=4) # To be updated by environment
    action_dim: int = field(default=4) # To be updated by environment
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    num_models_ensemble: int = 5
    model_update_freq: int = 10 # Update dynamics model every N real steps
    planning_horizon: int = 10 # For MPC/planning
    num_planning_candidates: int = 100 # For MPC/CEM
    elite_fraction: float = 0.1 # For CEM

@dataclass
class ManagerConfig:
    subgoal_dim: int = field(default=2) # To be updated by environment
    state_dim: int = field(default=4) # To be updated by environment
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    update_frequency_worker_steps: int = 100 # Manager updates every N worker steps
    goal_achievement_threshold: float = 0.05

@dataclass
class WorkerConfig:
    state_dim: int = field(default=4) # To be updated by environment
    action_dim: int = field(default=4) # To be updated by environment
    goal_dim: int = field(default=2) # To be updated by environment
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    discount_factor: float = 0.95
    replay_buffer_size: int = 100_000
    her_k: int = 4 # Number of hindsight goals to sample
    epsilon_start: float = 1.0 # For exploration (if applicable)
    epsilon_end: float = 0.05
    epsilon_decay: int = 50000 # Number of steps to decay epsilon

@dataclass
class EnvironmentConfig:
    env_name: str = "SimpleGridWorld" # or "FetchReach-v2", "MountainCarContinuous-v0"
    grid_size: int = 10 # For SimpleGridWorld
    max_episode_steps: int = 500 # Max steps per episode
    reward_scale: float = 1.0

@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    dynamics_model: DynamicsModelConfig = field(default_factory=DynamicsModelConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.general.results_dir, exist_ok=True)
        os.makedirs(self.general.pictures_dir, exist_ok=True)
