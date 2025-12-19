# CA24 Report

## Abstract

We present a concise reproducible experiment demonstrating how to map a simple regression problem to an MLP model with configurable components. The repository includes import-safe modules, YAML-based configs, and testing scaffolding to ensure reproducibility and ease of extension.

## Introduction

Outline the problem and the theoretical motivation. For this CA we use a synthetic linear regression task to validate the training pipeline and measurement logging.

## Methods

- Data: synthetic regression with known linear ground truth and additive Gaussian noise.
- Model: MLP with ReLU activations.
- Loss: Mean Squared Error (MSE), implemented via `WeightedMSE`.
- Optimization: Adam with configurable learning rate.

### Mathematical details

Given inputs x in R^d, target y in R, assume y = w^T x + epsilon where epsilon ~ N(0, sigma^2). Model approximates y_hat = f_theta(x). The MSE loss is L = E[(y - y_hat)^2].

## Experiments

1) Baseline: default config (see `configs/default.yaml`)
2) Hidden size ablation: try [32, 64, 128]
3) Learning rate ablation: try [1e-2, 1e-3, 1e-4]
4) Seed repeats: run each setting with 5 seeds and report mean ± std of final train loss.

## Results (template)

Include training loss curves and a results table summarizing final loss across settings. Example of the table in Markdown:

| Setting | Mean Final Loss | Std |
|---|---:|---:|
| baseline | 0.34 | 0.05 |
| hidden=32 | 0.40 | 0.06 |

## Reproducibility

- Record commands used and config files that generated each experiment.
- Save random seeds, hardware, and package versions.

## Conclusion

Summarize observation and future work directions.
