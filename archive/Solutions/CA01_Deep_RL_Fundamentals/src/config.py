from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DQNConfig:
    state_size: int = 4
    action_size: int = 2
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 64
    update_every: int = 4
    tau: float = 1e-3
    use_double_dqn: bool = False
    use_dueling: bool = False
    hidden_size: int = 64
    seed: int = 42

    def __post_init__(self):
        if not (0 < self.lr <= 1):
            raise ValueError("Learning rate must be between 0 and 1")
        if not (0 <= self.gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1")
        if not (0 <= self.epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1")


@dataclass
class REINFORCEConfig:
    state_size: int = 4
    action_size: int = 2
    lr: float = 1e-3
    gamma: float = 0.99
    hidden_size: int = 64
    seed: int = 42

    def __post_init__(self):
        if not (0 < self.lr <= 1):
            raise ValueError("Learning rate must be between 0 and 1")
        if not (0 <= self.gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1")


@dataclass
class ActorCriticConfig:
    state_size: int = 4
    action_size: int = 2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    gamma: float = 0.99
    hidden_size: int = 64
    seed: int = 42

    def __post_init__(self):
        if not (0 < self.lr_actor <= 1) or not (0 < self.lr_critic <= 1):
            raise ValueError("Learning rates must be between 0 and 1")
        if not (0 <= self.gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1")


@dataclass
class ExperimentConfig:
    env_name: str = "CartPole-v1"
    n_episodes: int = 1000
    max_t: int = 1000
    solve_threshold: float = 195.0
    num_runs: int = 3
    dqn_config: DQNConfig = DQNConfig()
    reinforce_config: REINFORCEConfig = REINFORCEConfig()
    actor_critic_config: ActorCriticConfig = ActorCriticConfig()


