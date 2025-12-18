# CA17 — Report

## Title
Simple Policy Gradient on CartPole-v1 (Baseline Implementation)

## Authors
Author: student@example.com

## Abstract
This report documents a minimal policy gradient implementation (Monte-Carlo / REINFORCE) used as a baseline for CA17. The implementation is intentionally small, import-safe, and designed for instructional clarity.

## Method
- Environment: CartPole-v1
- Policy: MLPPolicy (2-layer MLP, ReLU activations, softmax outputs)
- Learning rule: Policy gradient (REINFORCE) with entropy regularization
- Returns: Monte-Carlo discounted returns (no baseline except mean subtraction)

## Experimental Setup
- Seed: 42
- Optimizer: Adam, lr=1e-3
- Hidden size: 128
- Total timesteps: 50,000 (configurable via `src/config.py`)

## Results (Expected / Reproducibility)
This is a scaffolded assignment; full experimental runs should be carried out locally. Expect learning within 50k timesteps on CartPole-v1 with default settings; results will vary by seed and hardware.

## Discussion
- The training loop uses full-episode Monte-Carlo returns which are high-variance but simple to reason about.
- Improvements: use GAE, advantage normalization, mini-batch updates, multiple rollouts per update.

## How to reproduce
1. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
2. Install requirements: `pip install -r paperAssignments/Assignments1-50/CA17/requirements.txt`
3. Run training: `python -m paperAssignments.Assignments1-50.CA17.src.train` (or `python -m src.train` from the CA17 folder)
4. Run tests: `python -m pytest paperAssignments/Assignments1-50/CA17/tests`

## Files
- `src/` — implementation
- `tests/` — unit tests
- `notebooks/` — suggested place for experiments (not included by default)

## License
This assignment falls under the repository license. Any code submissions should follow the project's contribution guidelines.
