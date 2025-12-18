#!/usr/bin/env python3
"""
Compare bandit algorithms by calling the runner and saving results for plotting.

Usage:
    python compare_algorithms.py --num_arms 10 --steps 5000 --runs 50 --out_dir ../results
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

# Ensure we can import the runner in ../code
this_dir = os.path.dirname(__file__)
code_dir = os.path.abspath(os.path.join(this_dir, "..", "code"))
sys.path.insert(0, code_dir)

try:
    import run_bandits  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(f"Could not import run_bandits from {code_dir}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare bandit algorithms and save results"
    )
    parser.add_argument("--num_arms", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_dir", type=str, default="../results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = run_bandits.run_bandit_experiment(
        num_arms=args.num_arms, steps=args.steps, runs=args.runs, seed=args.seed
    )

    # Save results as compressed numpy archive for plotting
    out_path = os.path.join(args.out_dir, "bandit_results.npz")
    save_dict: Dict[str, np.ndarray] = {}
    for name, data in results.items():
        save_dict[f"{name}_rewards"] = data["rewards"]
        save_dict[f"{name}_regrets"] = data["regrets"]
    np.savez_compressed(out_path, **save_dict)
    print(f"Saved results to {out_path}")

    # Try to call the plotting helper if available
    viz_script = os.path.abspath(
        os.path.join(this_dir, "..", "visualize", "generate_plots.py")
    )
    if os.path.exists(viz_script):
        print("Calling visualization helper...")
        os.system(
            f'python "{viz_script}" --results "{out_path}" --out_dir "{args.out_dir}"'
        )
    else:
        print(f"Visualization script not found at {viz_script}; run it manually.")


if __name__ == "__main__":
    main()





