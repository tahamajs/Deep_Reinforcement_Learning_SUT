from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Hyperparameters and architecture configuration for TWM-SSD (Assignment CA11)."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    seq_len: int = 128
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-2
    dropout: float = 0.1
    ssm_ratio: float = 0.5  # fraction of blocks that are SSM/Mamba
    device: Optional[str] = None

def get_default_config() -> Config:
    return Config()

