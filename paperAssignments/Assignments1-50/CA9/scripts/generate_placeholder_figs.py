#!/usr/bin/env python3
"""
Generate placeholder PNG figures used by report.tex so LaTeX compiles cleanly.
Produces:
 - outputs/ca9/plots/losses.png
 - outputs/ca9/plots/lam_std.png
 - outputs/ca9/eval/returns.png
 - outputs/ca9/eval/ep_0/rewards.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_losses(path):
    steps = np.arange(0, 1000, 10)
    v = np.exp(-steps / 400.0) + 0.02 * np.random.randn(len(steps))
    c = np.exp(-steps / 300.0) + 0.02 * np.random.randn(len(steps))
    p = np.exp(-steps / 500.0) + 0.02 * np.random.randn(len(steps))
    plt.figure(figsize=(6, 3.5))
    plt.plot(steps, v, label="value loss")
    plt.plot(steps, c, label="critic loss")
    plt.plot(steps, p, label="policy loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_lam_std(path):
    steps = np.arange(0, 1000, 10)
    lam = 1.0 / (1.0 + np.exp((np.linspace(1.0, 0.01, len(steps)) - 0.3) * 12.0))
    std = np.linspace(0.5, 0.05, len(steps)) + 0.01 * np.random.randn(len(steps))
    plt.figure(figsize=(6, 3.5))
    plt.plot(steps, lam, label="Lambda")
    plt.plot(steps, std, label="Std")
    plt.xlabel("steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_returns(path):
    returns = [20, 22, 19, 23, 25]
    plt.figure(figsize=(6, 3.5))
    plt.plot(returns, marker="o")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_ep_rewards(path):
    steps = np.arange(0, 200)
    rews = np.sin(steps / 20.0) + 0.5 * np.random.randn(len(steps))
    plt.figure(figsize=(6, 3.5))
    plt.plot(steps, rews)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ensure_dir("outputs/ca9/plots")
    ensure_dir("outputs/ca9/eval/ep_0")
    make_losses("outputs/ca9/plots/losses.png")
    make_lam_std("outputs/ca9/plots/lam_std.png")
    make_returns("outputs/ca9/eval/returns.png")
    make_ep_rewards("outputs/ca9/eval/ep_0/rewards.png")
    print("Placeholder figures generated under outputs/ca9/")


if __name__ == "__main__":
    main()
