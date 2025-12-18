CA20 — Short Reproducible Report

Title: Lagrangian-constrained Policy Optimization — A Minimal Reproducible Study

Authors: Student Name

Abstract

This short report documents a minimal experiment using a simple MLP policy trained with a Lagrangian relaxation on a synthetic bandit-style dataset. The objective is to maximize reward while keeping mean constraint violations below a threshold c.

1. Method

We implement a Gaussian MLP policy with a state-independent log-std and a value network for baseline advantage estimation. The Lagrangian objective is

L() = E[policy loss] + mu * (C - c)

Where C is the mean batch constraint value and mu is a non-negative multiplier updated by projected gradient ascent.

2. Experimental Setup

- Synthetic dataset: 1000 samples, observations of dimension 8, actions of dimension 2
- Optimizer: Adam with lr from config
- Configs: `configs/debug.yaml` and `configs/default.yaml` used for debug and default runs.

3. Results (example)

Run a short debug training (2 epochs):

python -c "from paperAssignments.Assignments1_50.CA20.src import train, config; cfg=config.Config(); cfg.epochs=2; cfg.batch_size=32; train.train(cfg)"

Expected output: training history with policy/value/constraint entries and a saved checkpoint under `outputs/`.

4. Discussion

The synthetic dataset encourages nontrivial constraint violations correlated with a specific observation feature. The Lagrangian multiplier increases when constraint estimates exceed the threshold and reduces policy updates that lead to violations.

5. Reproducibility

- All code is import-safe and can be run via `python -m paperAssignments.Assignments1_50.CA20.src.train`.
- Config YAML files define parameters; `Config.from_yaml` loads them.
- Tests are provided under `tests/` and can be executed with `pytest -q`.

Appendix: Figures

Include reward/constraint curves saved by the demo notebook.
