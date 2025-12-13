
import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VAEConfig:
    obs_dim: int = 4  # Example for CartPole/Pendulum
    latent_dim: int = 32
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    num_episodes_data_collection: int = 100
    env_name: str = "Pendulum-v1"

@dataclass
class RSSMConfig:
    obs_dim: int = 4
    action_dim: int = 1  # Example for Pendulum-v1, will be updated dynamically
    latent_dim: int = 32
    hidden_dim: int = 256
    stochastic_size: int = 32
    learning_rate: float = 1e-3

@dataclass
class AgentConfig:
    latent_dim: int = 32
    action_dim: int = 1  # Example, will be updated dynamically
    hidden_dim: int = 256
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    imagination_horizon: int = 15
    gamma: float = 0.99
    batch_size: int = 50

@dataclass
class DreamerConfig:
    env_name: str = "Pendulum-v1"
    num_episodes: int = 1000
    max_steps: int = 200
    seed: int = 42
    vae_config: VAEConfig = VAEConfig()
    rssm_config: RSSMConfig = RSSMConfig()
    agent_config: AgentConfig = AgentConfig()

@dataclass
class GlobalConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    results_dir: str = "results"
    logs_dir: str = "logs"
    visualizations_dir: str = "visualizations"
    dreamer_config: DreamerConfig = DreamerConfig()
    vae_config: VAEConfig = VAEConfig()

# Initialize configurations
GLOBAL_CONFIG = GlobalConfig()
VAE_CONFIG = GLOBAL_CONFIG.vae_config
RSSM_CONFIG = GLOBAL_CONFIG.dreamer_config.rssm_config
AGENT_CONFIG = GLOBAL_CONFIG.dreamer_config.agent_config
DREAMER_CONFIG = GLOBAL_CONFIG.dreamer_config

def update_config_with_env_dims(env_name: str):
    """Dynamically update config with environment dimensions."""
    env = None
    if env_name == "Pendulum-v1":
        import gymnasium as gym
        env = gym.make(env_name)
    elif env_name == "continuous_cartpole":
        from environments.continuous_cartpole import ContinuousCartPole
        env = ContinuousCartPole()
    elif env_name == "continuous_pendulum":
        from environments.continuous_pendulum import ContinuousPendulum
        env = ContinuousPendulum()
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else 1
    env.close()

    VAE_CONFIG.obs_dim = obs_dim
    RSSM_CONFIG.obs_dim = obs_dim
    RSSM_CONFIG.action_dim = action_dim
    AGENT_CONFIG.action_dim = action_dim
