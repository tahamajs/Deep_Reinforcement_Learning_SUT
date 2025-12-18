"""
CA21 src package initializer.

Provides lightweight exports for convenience.
"""

from .config import Config, get_default_config  # noqa: F401
from .model import MLPPolicy, MLPValue  # noqa: F401
from .losses import policy_gradient_loss, value_mse_loss  # noqa: F401
from .data import SyntheticDataset  # noqa: F401
from .utils import set_seed, save_checkpoint, load_checkpoint  # noqa: F401

__all__ = [
    "Config",
    "get_default_config",
    "MLPPolicy",
    "MLPValue",
    "policy_gradient_loss",
    "value_mse_loss",
    "SyntheticDataset",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
]








