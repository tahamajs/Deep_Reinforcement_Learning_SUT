# Example Report — CA22 (Synthetic RL experiment)

Authors: Example Author

## Abstract

We evaluate a simple policy-gradient agent trained on a synthetic dataset with a constraint cost. The agent uses a Lagrangian formulation to penalize expected constraint violations. Results show the Lagrangian penalty effectively reduces the constraint mean at modest cost to reward.

## Methods

- Policy: 2-layer MLP producing logits (see `src/model.py`).
- Value: 2-layer MLP value baseline (see `src/model.py`).
- Loss: Policy gradient + Lagrangian penalty (see `src/losses.py`).
- Dataset: `SyntheticDataset` (see `src/data.py`) with short episodes.

## Experimental setup

- Config: `configs/debug.yaml` (seed=0, small hidden size for quick experiments).
- Optimizer: Adam, lr from config.
- Lagrange update: `src.utils.update_lagrange` used with lr from config.

## Results (example)

- Mean reward (baseline): 1.25 ± 0.15
- Mean constraint (Lagrangian): 0.48 ± 0.03 (target 0.5)

Figures and detailed tables should be placed under `outputs/figures/` with captions.

## Discussion

The Lagrangian penalty allowed controlling the expected constraint to near the threshold with a small reduction in reward. Future work could add learning-rate sweeps and compare to alternative constrained RL methods.

## Reproducibility

To reproduce:

1. Install dependencies: `pip install -r requirements.txt`
2. Run the demo notebook with `configs/debug.yaml`.
3. Use seed listed in the config; include generated `outputs/` and the final config in submission.


---

This is a short example to show the expected content; replace numbers with your own final results and include figures in `outputs/figures/`.