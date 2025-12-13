from dataclasses import dataclass, field
from typing import List

@dataclass
class BaseConfig:
    """Base configuration for all modules."""
    seed: int = 42
    device: str = "cuda" if "cuda" else "cpu" # Placeholder: will be updated by utils.setup
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005 # For soft target updates
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 256
    num_episodes: int = 500
    max_steps_per_episode: int = 200

@dataclass
class MultiAgentConfig(BaseConfig):
    """Configuration for multi-agent systems."""
    n_agents: int = 3
    state_dim: int = 10 # Placeholder: will be updated by environment
    action_dim: int = 4 # Placeholder: will be updated by environment
    coordination_mechanism: str = "centralized"  # centralized, decentralized, mixed

@dataclass
class CommunicationConfig(BaseConfig):
    """Configuration for emergent communication module."""
    message_dim: int = 32
    comm_lr: float = 1e-3 # Learning rate for communication module

@dataclass
class MAMLConfig(BaseConfig):
    """Configuration for Model-Agnostic Meta-Learning."""
    num_meta_episodes: int = 1000
    num_inner_steps: int = 1
    inner_lr_actor: float = 1e-2
    inner_lr_critic: float = 1e-2
    inner_lr_comm: float = 1e-2
    outer_lr_actor: float = 1e-3
    outer_lr_critic: float = 1e-3
    outer_lr_comm: float = 1e-3
    num_support_samples: int = 10 # K-shot learning
    num_query_samples: int = 10
    meta_batch_size: int = 4 # Number of tasks per meta-batch

@dataclass
class MCACConfig(MultiAgentConfig, CommunicationConfig, MAMLConfig):
    """Comprehensive configuration for the Meta-Communicative Actor-Critic framework."""
    # Inherits all settings from MultiAgentConfig, CommunicationConfig, and MAMLConfig
    pass


# Instantiate the main configuration object
config = MCACConfig()
