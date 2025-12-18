"""Small runnable demo script to run a short training and save outputs.

This script is intentionally minimal and meant to be executed manually by the user.
It does not run on import (import-safe).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.config import load_config
from src.utils import set_seed
from src.train import train_dqn


def main(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    # quick demo
    cfg.num_episodes = 50
    set_seed(cfg.seed)
    rewards = train_dqn(cfg)

    out = Path("outputs")
    figdir = Path("figures")
    out.mkdir(exist_ok=True)
    figdir.mkdir(exist_ok=True)

    np.save(out / "rewards.npy", np.array(rewards))

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training (demo)")
    plt.savefig(figdir / "training_curve.png")
    print(f"Saved rewards to {out / 'rewards.npy'} and figure to {figdir / 'training_curve.png'}")


if __name__ == "__main__":
    main()
