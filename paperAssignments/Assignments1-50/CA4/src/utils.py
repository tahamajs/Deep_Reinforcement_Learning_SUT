from typing import Iterable
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(
    target: Iterable[torch.nn.Parameter],
    source: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    for tp, sp in zip(target, source):
        tp.data.mul_(1.0 - tau)
        tp.data.add_(tau * sp.data)


def iqr_from_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Compute IQR along last-dimension for each batch row.
    x: [B, K]
    returns: [B]
    """
    q25 = x.kthvalue(max(1, int(0.25 * x.size(1))), dim=1)[0]
    q75 = x.kthvalue(max(1, int(0.75 * x.size(1))), dim=1)[0]
    return q75 - q25


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    # log_std: [B, a_dim]
    return 0.5 * (1.0 + torch.log(2 * torch.pi)) + log_std



