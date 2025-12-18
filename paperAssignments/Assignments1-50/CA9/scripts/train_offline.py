import argparse
import random

import numpy as np
import torch

from ..src.config import default_config
from ..src.algos.au_dmg import AUDMG


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    cfg = default_config()
    set_seed(cfg.seed)
    # NOTE: This script is a lightweight driver and does not download D4RL datasets.
    # It demonstrates how to instantiate the algorithm.
    s_dim = 10
    a_dim = 2
    agent = AUDMG(s_dim, a_dim, cfg)
    print(
        "Initialized AU-DMG agent. Ready to train (offline loop not implemented here)."
    )


if __name__ == "__main__":
    main()
