"""Script to generate example plots from CSV outputs (does not run here).

Usage:
    python scripts/plot_examples.py --rewards_csv results/ca31_rewards.csv --out_dir results/figures/

It requires matplotlib and numpy. The script loads per-step rewards and
outputs learning curve PNGs suitable for the LaTeX report.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(rewards, out_path):
    steps = np.arange(len(rewards))
    rewards = np.asarray(rewards)
    window = max(1, len(rewards)//100)
    # moving average
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    plt.figure(figsize=(8,4))
    plt.plot(steps, rewards, alpha=0.3, label='reward')
    plt.plot(steps[window-1:], smoothed, color='C0', label='smoothed')
    plt.xlabel('Environment steps')
    plt.ylabel('Episode reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewards_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV format: step,action,reward
    data = np.genfromtxt(args.rewards_csv, delimiter=',', names=True)
    if 'reward' in data.dtype.names:
        rewards = data['reward']
        plot_learning_curve(rewards, out_dir / 'learning_curve.png')
    else:
        raise RuntimeError('CSV does not contain reward column')
