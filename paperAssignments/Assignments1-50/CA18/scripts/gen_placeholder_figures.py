#!/usr/bin/env python3
"""Generate placeholder figures for CA18 report."""
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure pictures directory exists
os.makedirs('pictures', exist_ok=True)

# Fake data for reward curves
epochs = np.arange(1, 51)
seeds = 5
reward_data = []
for seed in range(seeds):
    np.random.seed(42 + seed)
    rewards = 10 + 5 * np.sin(epochs / 10) + np.random.normal(0, 1, len(epochs))
    rewards = np.cumsum(rewards) / np.arange(1, len(epochs)+1)  # cumulative average
    reward_data.append(rewards)

reward_mean = np.mean(reward_data, axis=0)
reward_std = np.std(reward_data, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(epochs, reward_mean, label='Mean Reward', color='blue')
plt.fill_between(epochs, reward_mean - reward_std, reward_mean + reward_std, alpha=0.3, color='blue')
plt.xlabel('Epoch')
plt.ylabel('Cumulative Reward')
plt.title('Reward Curves Across Seeds')
plt.legend()
plt.grid(True)
plt.savefig('pictures/fig_rewards.png', dpi=300, bbox_inches='tight')
plt.close()

# Fake data for loss curves
policy_loss = 2 * np.exp(-epochs / 20) + 0.1 * np.random.normal(0, 0.1, len(epochs))
value_loss = 1 * np.exp(-epochs / 15) + 0.05 * np.random.normal(0, 0.05, len(epochs))

plt.figure(figsize=(8, 6))
plt.plot(epochs, policy_loss, label='Policy Loss', color='red')
plt.plot(epochs, value_loss, label='Value Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.savefig('pictures/fig_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# Fake ablation data
entropy_coefs = [0.0, 0.01, 0.05, 0.1]
mean_returns = [12.5, 15.2, 14.8, 13.9]
std_returns = [3.2, 2.1, 2.5, 3.0]

plt.figure(figsize=(8, 6))
plt.errorbar(entropy_coefs, mean_returns, yerr=std_returns, fmt='o-', capsize=5)
plt.xlabel('Entropy Coefficient')
plt.ylabel('Mean Return')
plt.title('Ablation on Entropy Coefficient')
plt.grid(True)
plt.savefig('pictures/fig_ablation.png', dpi=300, bbox_inches='tight')
plt.close()

print("Placeholder figures generated in pictures/")
