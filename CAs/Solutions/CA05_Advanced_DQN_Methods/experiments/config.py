from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch


@dataclass
class AgentConfig:
    """Configuration for a single DQN agent."""
    agent_type: str = "dqn"
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 50000
    batch_size: int = 64
    target_update_freq: int = 500
    hidden_dim: int = 128
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 100000
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    env_name: str = "CartPole-v1"
    num_episodes: int = 1000
    seed: int = 42
    results_path: str = "./results"
    plots_path: str = "./visualizations"
    agent_config: AgentConfig = field(default_factory=AgentConfig)


def get_dqn_configs(env_name: str) -> Dict[str, AgentConfig]:
    """
    Returns a dictionary of AgentConfig objects for different DQN variants.

    Args:
        env_name (str): The name of the environment to configure for.

    Returns:
        Dict[str, AgentConfig]: A dictionary where keys are agent types
                                and values are their respective configurations.
    """
    base_config = AgentConfig(
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=500,
        hidden_dim=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    dqn_configs = {
        "dqn": AgentConfig(**base_config.__dict__, agent_type="dqn"),
        "double_dqn": AgentConfig(**base_config.__dict__, agent_type="double_dqn"),
        "dueling_dqn": AgentConfig(**base_config.__dict__, agent_type="dueling_dqn"),
        "prioritized_dqn": AgentConfig(
            **base_config.__dict__,
            agent_type="prioritized_dqn",
            priority_alpha=0.6,
            priority_beta_start=0.4,
            priority_beta_frames=200000,
        ),
        # Add Rainbow DQN config when implemented
        # "rainbow_dqn": AgentConfig(**base_config.__dict__, agent_type="rainbow_dqn"),
    }

    # Environment specific adjustments
    if env_name == "LunarLander-v2":
        for config in dqn_configs.values():
            config.target_update_freq = 1000
            config.buffer_size = 100000
            config.batch_size = 128
            config.lr = 5e-4
            config.epsilon_decay = 0.999

    return dqn_configs

