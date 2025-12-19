"""Generate placeholder figures used by the report template.

Creates `pictures/fig_rewards.png` and `pictures/fig_loss.png` as example artifacts.
"""
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "pictures"
OUT.mkdir(parents=True, exist_ok=True)

# Reward curve placeholder
x = np.linspace(0, 1000, 100)
y = np.log1p(x) + 0.1 * np.random.randn(len(x))
plt.figure(figsize=(6, 3))
plt.plot(x, y, label="Proposed")
plt.plot(x, np.log1p(x) - 0.05 * np.random.randn(len(x)), label="Baseline")
plt.xlabel("Steps")
plt.ylabel("Cumulative reward")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_rewards.png", dpi=150)
plt.close()

# Loss curve placeholder
x = np.linspace(0, 1000, 100)
policy_loss = np.exp(-x / 400.0) + 0.02 * np.random.randn(len(x))
value_loss = np.exp(-x / 500.0) + 0.02 * np.random.randn(len(x))
plt.figure(figsize=(6, 3))
plt.plot(x, policy_loss, label="policy loss")
plt.plot(x, value_loss, label="value loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_loss.png", dpi=150)
plt.close()

print("Wrote:", list(OUT.iterdir()))
