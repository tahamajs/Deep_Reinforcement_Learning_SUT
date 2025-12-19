"""CA31 package — small, import-safe utilities for the CA31 assignment.

The package contains a reproducible multi-armed bandit example (numpy-only) with
config loading and seeding helpers for deterministic experiments.
"""

__all__ = ["utils", "bandit", "train"]

from . import utils, bandit, train
