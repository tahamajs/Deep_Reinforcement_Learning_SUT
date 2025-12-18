from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseMuZeroPolicy(ABC):
    """Minimal local base policy class mimicking LightZero/ma_muzero policy API.

    This base class exists so concrete adapters in this repo can inherit a stable interface
    expected by LightZero-like training loops.
    """

    @abstractmethod
    def infer(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def search(self, obs: torch.Tensor, sims: int = 100, topk: int = 8) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor], loss_weights: Dict[str, float], optimizer: Any) -> Any:
        ...

    def save(self, path: str) -> None:
        """Optional: save policy weights."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Optional: load policy weights."""
        raise NotImplementedError

