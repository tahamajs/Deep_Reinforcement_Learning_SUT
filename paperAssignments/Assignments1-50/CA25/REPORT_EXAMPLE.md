# Example Report — CA25 Example Run

**Title:** Toy MLP on Synthetic Regression

**Authors:** Student Name

**Abstract**
We run a short experiment training a small MLP on a synthetic regression dataset to validate the training loop and reproducibility infrastructure. The model converges in a few epochs and reproduces consistent loss curves across seeds.

## 1 Introduction
This brief experiment demonstrates a simple supervised learning pipeline with a configurable MLP and synthetic data. The goal is to verify reproducible training and the artifact saving pipeline.

## 2 Methods
Model: 2 hidden layers [64, 32] with ReLU activations; final linear output. See `src/model.py`.

Data: Synthetic linear regression with Gaussian noise; 1000 samples, 16 features. See `src/data.py`.

Training: Adam optimizer with lr=1e-3, batch size 128, mean-squared error loss. Config used: `configs/example.yaml`.

## 3 Experimental Setup
Single run with seed 123, 5 epochs (short smoke test). No hyperparameter sweep performed.

## 4 Results
Loss curve saved to `outputs/<run>/pictures/loss.png` (example placeholder). Final validation loss (example): 0.08.

> Replace the above numbers with actual metrics from your run.

## 5 Discussion
This smoke test confirms the training pipeline, artifact saving, and that the unit tests pass. For a full study, run multiple seeds and report mean ± std.

## 6 Reproducibility
Run the training command with the example config to reproduce this example, then copy plots and values into the report.

```bash
python -m paperAssignments.Assignments1-50.CA25.train --config configs/example.yaml
```
