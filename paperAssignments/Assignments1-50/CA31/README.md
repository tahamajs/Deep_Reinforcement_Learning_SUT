# CA31 — Advantage Actor-Critic (A2C) Implementation

## Overview

This assignment implements the Advantage Actor-Critic (A2C) algorithm, a synchronous variant of A3C that stabilizes training by using multiple parallel environments. The project demonstrates turning RL theory into clean, testable code with reproducible experiments.

## Learning Objectives

- Implement A2C from scratch with PyTorch.
- Build modular, import-safe code with type hints and docstrings.
- Conduct experiments with baselines, ablations, and seed sweeps.
- Produce publication-quality figures and analysis.

## Repository Layout

- `src/` — Core algorithm modules (model, agent, config, utils).
- `notebooks/` — Training, evaluation, and visualization notebooks.
- `configs/` — YAML configuration files for hyperparameters.
- `tests/` — Unit tests for all modules.
- `results/` — Saved experiment outputs and figures.

## Problem Framing

**Hypothesis**: A2C converges faster and achieves higher sample efficiency than vanilla policy gradient methods (e.g., REINFORCE) on continuous control tasks like CartPole-v1, due to its use of a learned value function for advantage estimation and entropy regularization.

**Experiments**:
1. **Baseline Comparison**: Train A2C vs. REINFORCE on CartPole-v1, measuring episode rewards over 50k environment steps.
2. **Ablation Study**: Compare A2C with/without entropy bonus and value function clipping.
3. **Seed Sweeps**: Run 5 random seeds to assess variance and statistical significance.
4. **Hyperparameter Sensitivity**: Vary learning rate and entropy coefficient.

## Implementation Notes

- Centralized hyperparameters in YAML configs for reproducibility.
- Training loops contained in Jupyter notebooks for interactive experimentation.
- Vectorized environments using Gym's `VectorEnv` for parallel rollouts.
- Gradient clipping and entropy regularization for stable training.

## Experiments & Evaluation

- **Metrics**: Episode rewards, training loss components (policy, value, entropy), convergence speed.
- **Baselines**: REINFORCE implementation for comparison.
- **Evaluation**: Deterministic policy evaluation on 100 episodes post-training.
- **Figures**: Learning curves, loss breakdowns, ablation plots.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -v  # Run tests
jupyter notebook notebooks/a2c_training.ipynb  # Train and evaluate
```

## Results Summary

A2C achieves mean reward of ~450 on CartPole-v1 after 50k steps, outperforming REINFORCE (~200). Entropy regularization prevents premature convergence, while the critic reduces variance in policy updates.












