# CA21 — Short Report

## Objective

This mini-project demonstrates a minimal, reproducible training loop for a discrete-action stochastic policy and a value function trained on synthetic data. The aim is educational: provide a scaffold for experiments such as regularized policy optimization, baseline comparisons, and seed sensitivity analyses.

## Implementation Summary

- `src/model.py`: Small MLP backbone (`MLPBase`) used by `MLPPolicy` and `MLPValue`.
- `src/data.py`: `SyntheticDataset` returning (obs, action, reward, next_obs, done) tuples.
- `src/losses.py`: `policy_gradient_loss` (negative expected value of log-prob * advantage) and `value_mse_loss` (MSE).
- `src/utils.py`: Seeding and checkpoint helpers.
- `src/train.py`: Minimal end-to-end training loop with a simple advantage estimate (reward - value) used for the demo.

## Experiments and Results

All experiments below are quick debug runs intended to be reproducible in CI or locally using the `configs/debug.yaml` settings.

- Smoke training: `python -m src.train --epochs 2 --batch-size 8`
  - Expected: training completes without errors and produces a metrics dictionary with `final_pg_loss`, `final_value_loss`, and `seconds`.

- Unit tests: `pytest -q`
  - Tests cover import safety, forward-pass shapes, loss behavior, and checkpoint save/load.

## Reproducibility

- Seed control: `src.utils.set_seed` sets `random`, `numpy`, and `torch` seeds.
- Config centralization: `src.config.Config` holds hyperparameters; changeable via code or YAML configs used in notebooks.
- Checkpointing: `src.utils.save_checkpoint` / `load_checkpoint` save and load training state.

## How to extend

- Replace `SyntheticDataset` with environment rollouts or logged trajectories.
- Add a proper advantage estimator (GAE) and baseline updates.
- Implement evaluation loops with deterministic policy action selection and logging of episode returns.

## Files of interest

- `src/train.py` — training loop
- `src/losses.py` — learning objectives
- `notebooks/demo.ipynb` — quick interactive demo using `configs/debug.yaml`

## Notes and limitations

- This demo uses synthetic data and a very simple advantage signal for instructional purposes; it is not intended to be a research baseline out-of-the-box.
- Device selection prefers CUDA when available; tests run on CPU by default.

