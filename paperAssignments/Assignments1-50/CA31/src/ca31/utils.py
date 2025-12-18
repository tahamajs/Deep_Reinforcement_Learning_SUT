"""Utility functions: config loading and deterministic seeding.

Design goals:
- Keep imports small and stable (no heavy runtime dependencies).
- Use numpy generators for per-run determinism.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import yaml
import numpy as np


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return as a dictionary.

    Args:
        path: Path to a YAML file.
    Returns:
        A dict with configuration values.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: Optional[int]) -> np.random.Generator:
    """Set seeds for RNGs and return a numpy Generator.

    This helper centralizes seeding so experiments are reproducible. It sets
    Python's `random`, the `PYTHONHASHSEED` environment variable, and returns
    a `numpy.random.Generator` seeded deterministically.
    """
    if seed is None:
        seed = np.random.SeedSequence().entropy
    # int cast for env vars
    seed_int = int(seed) & 0xFFFFFFFF

    os.environ["PYTHONHASHSEED"] = str(seed_int)
    random.seed(seed_int)
    # Use PCG64 for reproducibility and explicit generator passing
    rng = np.random.default_rng(seed_int)
    return rng
