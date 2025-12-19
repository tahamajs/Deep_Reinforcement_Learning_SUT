#!/usr/bin/env python3
"""Simple plotting utility for CA24 outputs.

Loads `outputs/` produced by `scripts/run_sweep.py` and generates summary figures.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/sweep")
FIGDIR = Path("outputs/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

settings = [p for p in OUT.iterdir() if p.is_dir()]
summary = []
for s in settings:
    values = []
    for run in s.glob("seed-*/summary.json"):
        data = json.loads(run.read_text())
        values.append(data["final_train_loss"])
    if values:
        summary.append((s.name, float(np.mean(values)), float(np.std(values))))

if summary:
    names, means, stds = zip(*summary)
    x = range(len(names))
    plt.figure(figsize=(8, 4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Final train loss")
    plt.tight_layout()
    outp = FIGDIR / "summary.png"
    plt.savefig(outp, dpi=200)
    print(f"Saved summary figure to {outp}")
else:
    print("No results found in outputs/sweep. Run the sweep first.")
