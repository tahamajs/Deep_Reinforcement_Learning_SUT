from dataclasses import dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class Config:
    # General
    seed: int = 42
    device: str = "cpu"

    # C51
    c51_num_atoms: int = 51
    c51_v_min: float = -10.0
    c51_v_max: float = 10.0

    # QR-DQN
    qr_num_quantiles: int = 51
    qr_kappa: float = 1.0

    # Rainbow defaults
    rainbow_n_steps: int = 3
    rainbow_num_atoms: int = 51
    rainbow_v_min: float = -10.0
    rainbow_v_max: float = 10.0

    # TD3 defaults
    td3_gamma: float = 0.99
    td3_tau: float = 0.005
    td3_policy_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_policy_delay: int = 2

    # TRPO defaults
    trpo_max_kl: float = 1e-2
    trpo_damping: float = 1e-1
    trpo_cg_iters: int = 10
    trpo_backtrack_iters: int = 10
    trpo_backtrack_coeff: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


CFG = Config()







