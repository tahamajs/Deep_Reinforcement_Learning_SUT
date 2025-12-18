# CA22 — Curriculum Assignment 22

## Overview

CA22 is an assignment scaffold intended for research-style projects. This README outlines objectives, theoretical grounding, implementation patterns, evaluation practices, and expected deliverables. The document mirrors the long-format style used across other CAs and is suitable as a lecture-style brief.

## Learning Outcomes

- Implement a reproducible research pipeline.
- Translate mathematical derivations into stable, testable code.
- Conduct rigorous experiments with proper randomization and logging.
- Produce high-quality visualizations and a short written report.

## Repository Layout

- `src/` — core modules (import-safe): `config.py`, `model.py`, `losses.py`, `data.py`, `utils.py`.
- `notebooks/` — experiment notebooks that save outputs to `pictures/`.
- `configs/` — YAML/JSON configuration sets.
- `tests/` — pytest-compatible tests for core modules.
- `outputs/` — for runtime artifacts (not committed).

## Conceptual Focus

Pick a specific conceptual focus for CA22 (e.g., robust RL, imitation learning, or constrained optimization). The README provides a template for deriving objectives and mapping them to implementation.

Notation & example objective

- Let \pi\_\theta(a|s) be a stochastic policy.
- The target objective is to maximize expected return under a constraint C(\pi) \leq c (e.g., safety constraint).
- Use Lagrangian relaxation to form augmented objective:
  L(\theta, \mu) = E*{\tau\sim\pi\theta}[\sum_t \gamma^t r_t] - \mu (E*{\tau}[C(\tau)] - c)

Mapping to code

- Implement Lagrange multiplier updates in `src/utils.py` or `src/losses.py` with clear interfaces.
- Provide safeguards for multiplier stability (clipping, learning rate scheduling).

## Implementation Guidelines

- Use dataclasses for config management and YAML loader for user-specified overrides.
- Use helper functions to centralize device placement.
- Ensure robust checkpointing that saves both model and optimizer states.

Debug & default configs

- `debug`: small batch sizes, few epochs, deterministic seeds.
- `default`: scaled-up config for main experiments.

Notebook guidance

- Notebook should demonstrate full pipeline using `debug` config.
- Notebook must include code cells for: imports & setup, quick data preview, model instantiation, short training loop, evaluation, and plotting.
- All figures must be saved programmatically to `pictures/` using relative paths.

Experiments
Minimum experiments to perform:

- Baseline vs method with constraint.
- Sensitivity to constraint threshold c.
- Stability across seeds.

Evaluation & metrics

- Primary metric: average episode return under constraint adherence.
- Constraint metric: fraction of episodes violating constraint.
- Report both mean and standard deviation across seeds.

Visualization

- Reward and constraint violation curves over training steps.
- Scatter plots of policy behavior metrics (e.g., mean action magnitude vs reward).

Tests

- Unit tests for forward passes and loss function outputs.
- Integration smoke test that runs a short debug training loop (guarded) and verifies outputs are generated.

Deliverables

- `src/` modules implementing the algorithm.
- `notebooks/demo.ipynb` demonstrating a debug run and saving figures.
- `configs/` with `debug.yaml` and `default.yaml`.
- `tests/` with at least three tests.
- `README.md` (this file) describing usage and hyperparameters.

Appendix: Implementation checklist

1. Draft the mathematical derivation and map to code functions.
2. Implement `src/config.py` with dataclass and YAML loader.
3. Implement `src/model.py` with network modules and forward signature.
4. Implement `src/losses.py` for Lagrangian losses.
5. Implement `src/data.py` with small synthetic data generator.
6. Implement `src/utils.py` for seeding and checkpointing.
7. Add `notebooks/demo.ipynb` with the debug pipeline.
8. Add tests and run `pytest` for touched modules.
9. Save figures to `pictures/`.
10. Create a brief `report.tex` (optional).

Detailed Appendix: Micro-tasks (padding section)
