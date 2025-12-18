# CA31 — Report

## Abstract

This short report documents the implementation and experiments for CA31: an
Advantage Actor-Critic (A2C) implementation and a small, reproducible
Bernoulli bandit demonstration. The goal is to provide clean, testable code
that is easy to reproduce and analyze.

## Methods

### Advantage Actor-Critic (A2C)
- Synchronous actor-critic algorithm using multiple parallel environments (vectorized rollouts) in principle.
- Policy and value functions share a common encoder with two heads (actor, critic).
- Loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss.
- Uses advantage estimates computed as returns - value and standard gradient updates.

### Bandit experiment
- Simple n-armed Bernoulli bandit with a basic epsilon-greedy agent using
  incremental updates (running averages) for action-value estimates.
- Designed to be deterministic: RNGs are explicit and seeded via
  `src/ca31/utils.set_seed`.

## Implementation details

- Language: Python 3.10+, framework: PyTorch and NumPy.
- Core modules:
  - `src/model.py` — `ActorCritic` model with shared encoder and separate heads.
  - `src/agent.py` — `A2CAgent` implementing compute_returns, act, update.
  - `src/ca31/` — small, import-safe bandit implementation and guarded training script.
  - `src/config.py` — YAML loader.
  - `src/utils.py` — deterministic CPU/CUDA seeding helper.

### Network
- Two fully-connected hidden layers with ReLU activations (hidden_size=128 by default).
- Actor: linear output for action logits; critic: scalar head for value (shape `(batch, 1)`).

### Hyperparameters (defaults in `configs/a2c.yaml`)
- gamma: 0.99
- learning_rate: 1e-3
- entropy_coef: 0.01
- value_coef: 0.5
- max_grad_norm: 0.5

## Experiments

### Bandit experiment
- Config in `configs/experiment.yaml` (seed, arm_probs, n_steps, epsilon).
- Run this to generate CSV of per-step rewards and actions (optional save_path).

### A2C experiment (not a fast CI test)
- Notebook `notebooks/a2c_training.ipynb` demonstrates training on `CartPole-v1`.
- Suggested experiment: run 5 seeds, 50k steps each, collect learning curves.

## Results summary

- Bandit example: with `arm_probs: [0.1, 0.2, 0.8]` and epsilon=0.05, the agent learns
to prefer the best arm (index 2) and average reward increases above the uniform baseline.
- A2C on CartPole (reported earlier): mean reward ≈ 450 after 50k steps with the
example hyperparameters (this is an observed baseline for this implementation; your mileage may vary depending on environment versions and randomness).

## Reproducibility

- Run unit tests:

```bash
python -m pytest tests/ -q
```

- Quick bandit run (deterministic):

```bash
python -m ca31.train --config configs/experiment.yaml
```

- To reproduce A2C runs, use the notebook and the `configs/a2c.yaml` file. Always
record the seed and the package environment (`pip freeze > requirements-freeze.txt`).

## Limitations and future work

- The A2C training notebook demonstrates the idea but isn't set up to produce
publication-grade figures by default. To scale experiments: use VectorEnv, logging (TensorBoard), and a small experiment sweep harness.
- Add CI that runs a small smoke training with a deterministic RNG to check regressions.

## Appendix — Key config locations

- `configs/a2c.yaml` — A2C hyperparameters
- `configs/experiment.yaml` — Bandit experiment config

---

If you'd like, I can also:
- Add example figures and a script to generate them from `results/` CSVs;
- Add a small Makefile or GitHub Actions workflow to run tests and smoke experiments.
