from dataclasses import dataclass
import yaml

@dataclass
class Config:
    seed: int = 42
    learning_rate: float = 0.001
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update: int = 10
    memory_size: int = 10000
    num_episodes: int = 500
    max_steps: int = 200
    env_name: str = "CartPole-v1"
    double_dqn: bool = False
    replay: str = "uniform"  # options: 'uniform', 'prioritized'
    replay_alpha: float = 0.6
    replay_beta: float = 0.4

def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)