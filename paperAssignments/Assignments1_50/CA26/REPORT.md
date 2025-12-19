# CA26 — Project Report

## Abstract

This project implements a small, reproducible synthetic regression benchmark to evaluate different regression losses and small MLP architectures. The goal is to provide a clean research scaffold: configuration-driven experiments, unit-tested components, and an example training script that saves models and figures.

## Introduction

The CA26 assignment demonstrates the practice of turning a concise algorithmic idea into a reproducible experimental pipeline. The dataset is synthetic (sinusoidal function with noise), the models are small MLPs, and the primary objective is to compare loss functions and training stability.

## Method

- Dataset: y = sin(2πx) + ε, x ∈ [0,1]. Gaussian observation noise with configurable std.
- Models: Fully-connected MLP with configurable hidden widths and activation.
- Losses: Mean Squared Error (MSE) and Huber loss (smooth L1).
- Training: Adam optimizer, deterministic seeding, artifact saving (model state dict and loss curve).

### Implementation mapping
- Data: `src/data.py` provides `SyntheticRegressionDataset` and `get_dataloader`.
- Model: `src/model.py` contains `MLP` and `get_activation` helper.
- Losses: `src/losses.py` implements `mse_loss` and `huber_loss` with shape checks.
- Training: `scripts/run_experiment.py` exposes `fit(cfg, out_dir)` and a `main()` CLI.
- Utilities: `src/utils.py` provides deterministic seeding and plotting helpers.

## Experiments

Run a baseline experiment with the provided config:

```bash
python -m scripts.run_experiment --config configs/default.yaml --out outputs
```

This will create `outputs/model.pt` and `outputs/loss/loss_curve.png`.

### Suggested experimental protocol
- Compare MSE vs Huber by changing the loss function in `scripts/run_experiment.py`.
- Run each configuration for multiple seeds and report mean ± std of final loss.
- Keep experiments small for CI (epochs ≤ 3) and larger for final results (epochs ≥ 30).

## Results (example, replace with real runs)

| Model | Loss | Epochs | Final loss (mean ± std) |
|---|---:|---:|---:|
| MLP (64,64) | MSE | 30 | 0.012 ± 0.003 |
| MLP (64,64) | Huber (δ=1.0) | 30 | 0.010 ± 0.002 |

Note: Replace the table with numbers from your actual runs. The repo includes `save_loss_curve` which writes `loss_curve.png` for quick visual inspection.

## Reproducibility

- Virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt  # requirements listed in README
```

- Run tests:

```bash
python -m pytest -q
```

- Reproduce experiment:

```bash
python -m scripts.run_experiment --config configs/default.yaml --out outputs
```

## Discussion & Future Work

- Add experiments comparing training stability across noise levels and model capacity.
- Add automatic hyperparameter sweep (hydra, simple grid, or sweep scripts).
- Add more rigorous evaluation (RMSE, calibration, uncertainty estimates).

## Appendix - How to add figures

- Figures are generated programmatically via `src/utils.save_loss_curve`. For publication, gather outputs across seeds and create aggregated plots in a small notebook under `notebooks/`.

## Credits & License

Author: CA26 scaffold
License: MIT
