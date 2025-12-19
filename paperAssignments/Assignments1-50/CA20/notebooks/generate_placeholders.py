#!/usr/bin/env python3
"""
Generate placeholder figures for CA20 report.
Run this to create dummy plots under notebooks/pictures/ so the LaTeX report compiles.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("notebooks/pictures", exist_ok=True)

# Generate dummy reward curve
steps = np.arange(0, 100, 5)
rewards = 0.5 + 0.3 * np.sin(steps / 10) + 0.1 * np.random.randn(len(steps))
plt.figure(figsize=(8, 6))
plt.plot(steps, rewards, label="Reward")
plt.xlabel("Training Steps")
plt.ylabel("Average Reward")
plt.title("Placeholder Reward Curve")
plt.legend()
plt.grid(True)
plt.savefig("notebooks/pictures/reward_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Generate dummy constraint curve
constraints = 0.2 - 0.15 * np.exp(-steps / 20) + 0.05 * np.random.randn(len(steps))
plt.figure(figsize=(8, 6))
plt.plot(steps, constraints, label="Constraint Violation", color="red")
plt.xlabel("Training Steps")
plt.ylabel("Mean Constraint Violation")
plt.title("Placeholder Constraint Violation Curve")
plt.legend()
plt.grid(True)
plt.savefig("notebooks/pictures/constraint_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print("Placeholder figures generated in notebooks/pictures/")