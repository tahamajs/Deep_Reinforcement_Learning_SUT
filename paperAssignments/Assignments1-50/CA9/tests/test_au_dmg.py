import math
import torch
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ..src.config import default_config  # type: ignore
from ..src.algos.au_dmg import AUDMG  # type: ignore


def test_update_shapes_and_nans():
    cfg = default_config()
    s_dim = 6
    a_dim = 2
    agent = AUDMG(s_dim, a_dim, cfg)
    B = 4
    batch = {
        "s": torch.randn(B, s_dim),
        "a": torch.randn(B, a_dim),
        "r": torch.randn(B),
        "s_next": torch.randn(B, s_dim),
        "done": torch.zeros(B),
    }
    stats = agent.update(batch)
    assert "v_loss" in stats and "critic_loss" in stats and "policy_loss" in stats
    # values finite
    for v in stats.values():
        assert math.isfinite(v)


def test_gate_monotonicity_formula():
    cfg = default_config()
    k = cfg.kappa
    b = cfg.beta

    def gate(std):
        return 1.0 / (1.0 + math.exp(-((k / (std + 1e-6)) - b)))

    stds = [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]
    gates = [gate(s) for s in stds]
    # gate should be non-increasing as std increases
    for i in range(len(gates) - 1):
        assert gates[i] >= gates[i + 1]










