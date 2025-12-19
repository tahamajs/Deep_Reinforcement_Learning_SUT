"""Generate synthetic example results and an example figure (does not run training).

Run: python scripts/generate_synthetic_results.py
This will save:
 - outputs/synthetic_rewards_baseline.npy
 - outputs/synthetic_rewards_double.npy
 - outputs/synthetic_rewards_prioritized.npy
 - figures/example_training_curve_generated.png

The script is purely demonstrational and produces deterministic synthetic curves.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("outputs")
FIG = Path("figures")
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

np.random.seed(0)
episodes = 500
x = np.arange(1, episodes + 1)

# synthetic learning curves (smoothed)
baseline = 200 - 250 * np.exp(-x / 80) + np.random.randn(episodes) * 2
double = baseline + np.clip(np.linspace(0, 5, episodes) + np.random.randn(episodes) * 1, -2, 8)
prior = baseline + np.clip(np.linspace(0, 6, episodes) + np.random.randn(episodes) * 1, -2, 9)

# Save arrays
np.save(OUT / "synthetic_rewards_baseline.npy", baseline)
np.save(OUT / "synthetic_rewards_double.npy", double)
np.save(OUT / "synthetic_rewards_prioritized.npy", prior)

# Plot and save
plt.figure(figsize=(8, 4))
plt.plot(x, baseline, label="Baseline", color="#1f77b4")
plt.plot(x, double, label="Double DQN", color="#2ca02c")
plt.plot(x, prior, label="Prioritized Replay", color="#ff7f0e")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Synthetic Example Training Curves")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "example_training_curve_generated.png", dpi=150)
print(f"Saved synthetic curves to {OUT} and {FIG}")
