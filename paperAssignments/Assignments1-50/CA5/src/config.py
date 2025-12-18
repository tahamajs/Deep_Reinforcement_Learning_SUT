from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    env: str = "antmaze-medium-diverse-v2"
    gamma: float = 0.995
    c: int = 10
    hidden: int = 1024
    batch: int = 512
    bn_momentum: float = 0.01
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    entropy_beta: float = 0.05
    device: Optional[str] = "cpu"


def default_config():
    return Config()










