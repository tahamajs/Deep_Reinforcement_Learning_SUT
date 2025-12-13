from dataclasses import dataclass

@dataclass
class WorldModelConfig:
    """
    Configuration for the World Model.
    """
    observation_shape: tuple = (3, 64, 64)  # Example for visual observations
    action_dim: int = 1  # Placeholder, will be set by environment
    latent_dim: int = 32  # Dimension of stochastic latent state (s_t)
    hidden_dim: int = 256  # Dimension of recurrent hidden state (h_t)
    feature_dim: int = 1024 # Dimension of features before dynamics model
    rssm_layers: int = 1
    rssm_type: str = 'discrete' # 'discrete' or 'continuous'
    kl_loss_scale: float = 0.1
    reward_clip: float = 1.0
    free_nats: float = 0.0 # Free bits for KL divergence

@dataclass
class ManagerConfig:
    """
    Configuration for the Manager (high-level policy).
    """
    goal_dim: int = 32 # Dimension of the goal vector
    goal_horizon: int = 10 # Number of worker steps per manager step (H_M)
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    num_critics: int = 2 # For ensemble critics

@dataclass
class WorkerConfig:
    """
    Configuration for the Worker (low-level policy).
    """
    sub_goal_horizon: int = 5 # Number of environment steps per worker sub-episode (N)
    intrinsic_reward_weight: float = 0.5 # Alpha (alpha)
    extrinsic_reward_weight: float = 0.5 # (1 - alpha)
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    num_critics: int = 2 # For ensemble critics
    exploration_amount: float = 0.3 # For exploration noise

@dataclass
class TrainingConfig:
    """
    Configuration for the overall training procedure.
    """
    batch_size: int = 50
    sequence_length: int = 50 # Length of sequences sampled for world model training
    replay_buffer_size: int = 1_000_000
    environment_steps: int = 1_000_000 # Total environment steps
    model_retain_steps: int = 1_000_000 # Steps to keep model in buffer
    train_every_steps: int = 100 # Train models every N environment steps
    seed: int = 42
    log_interval: int = 1000 # Log every N steps
    eval_interval: int = 10000 # Evaluate every N steps
    checkpoint_interval: int = 50000 # Save checkpoint every N steps
    gradient_clip_norm: float = 100.0

@dataclass
class EnvironmentConfig:
    """
    Configuration for the environment.
    """
    env_name: str = 'MiniGrid-DoorKey-8x8-v0' # Default environment
    action_repeat: int = 1
    image_size: tuple = (64, 64) # For visual environments
    grayscale: bool = False
    reward_scale: float = 1.0 # Scale rewards
    time_limit: int = 1000 # Max steps per episode

