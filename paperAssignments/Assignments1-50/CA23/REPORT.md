# CA23 — Report

## Abstract

This short report documents the CA23 scaffold: design decisions, proposed
experiments, and reproducibility instructions. The repository provides simple
policy and value networks, loss utilities for policy-gradient algorithms, and
minimal data utilities for on-policy rollouts. The goal is to enable quick
experiment iteration and reproducible reporting for course projects.

## Goals and Hypothesis

- Goal: Evaluate a baseline policy-gradient agent against a simple environment
  benchmark and investigate the effect of entropy regularization and
  network capacity on learning stability.
- Hypothesis: Moderate entropy regularization improves exploration in early
  training and leads to more stable returns across seeds; larger networks
  reduce bias at the cost of increased variance.

## Methods / Implementation

- Networks: `PolicyNetwork` (categorical) and `ValueNetwork` (scalar) in
  `src/model.py` with configurable hidden sizes.
- Losses: standard policy gradient (`policy_gradient_loss`), MSE value loss
  (`value_loss`), and entropy term (`entropy_loss`) in `src/losses.py`.
- Data: `collect_episodes` for rolling out policies and `discounts` for return
  computation in `src/data.py`.
- Utilities: seeding (`set_seed`), device selection (`get_device`), and
  figure saving (`save_figure`).

## Suggested Experiments

1. Baseline training on `CartPole-v1` using `ExperimentConfig` default values.
2. Entropy sweep: train with entropy coefficients {0.0, 0.01, 0.1} and compare
   mean returns across 5 seeds.
3. Network size ablation: hidden sizes `[(32,32), (64,64), (128,128)]`.
4. Learning rate sensitivity: sweep {1e-4, 1e-3, 3e-3}.

For each run, record per-episode return, loss terms, and optionally save
checkpoints and figures.

## Evaluation Metrics & Reporting

- Primary metric: average episode return (smoothed by moving average, e.g.,
  window=50).
- Secondary metrics: policy/value losses, entropy, and sample efficiency
  (episodes to threshold).

Suggested figures:
- Learning curves (return vs. episodes) with shaded error over seeds.
- Bar charts comparing final performance across hyperparameters.

Suggested tables:
- Mean ± std return at convergence for each hyperparameter setting.

## Reproducibility Checklist

- Record the exact YAML config used (store under `runs/<id>/config.yaml`).
- Fix RNG seeds (call `set_seed` at start of training) and record seeds.
- Log random seeds and software/hardware environment (Python, torch, CPU/GPU).

## How to extend

- Add Advantage estimators (GAE), baselines, or actor-critic loops into
  `scripts/train.py` or dedicated notebooks.
- Add more robust evaluation (e.g., multiple environments and deterministic
  evaluation runs).

## Notes & Caveats

- This scaffold purposely keeps dependencies minimal and keeps heavy runtime
  out of imports so unit tests remain fast.
- For research-grade experiments, add checkpointing, structured logging
  (e.g., CSV/JSONL), and a more detailed experiment runner.

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction" (for PG/advantage).
- OpenAI Spinning Up: implementation references and best practices.
