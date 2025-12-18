# CA24 — Curriculum Assignment 24 ✅

## Overview

CA24 is a short reproducible experiment template designed to teach how to map theory to tidy, import-safe code. The project includes:

- A small, import-safe Python package under `src/` with dataclass-driven config, model, data, losses, utilities, and an experiment runner ✅
- A `notebooks/demo.ipynb` showing how to run the experiment without executing heavy training ✅
- Tests and GitHub Actions to verify import-safety and basic behaviour ✅

---

## Quick Start 🔧

1. Create a virtual environment and activate it:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run the demo experiment (CPU-friendly):

```bash
python -m src.experiment
```

3. Run tests:

```bash
pytest -q
```

---

## Project Layout 📁

- `src/` — core package
  - `config.py` — dataclass `Config` and `load_from_yaml`
  - `model.py` — `SimpleMLP` network
  - `data.py` — `SyntheticRegressionDataset` and `get_dataloader`
  - `losses.py` — `WeightedMSE` example loss
  - `utils.py` — `set_seed`, `get_device`, helpers
  - `experiment.py` — `run_experiment` and a small CLI demo
- `configs/` — YAML configs (see `default.yaml`)
- `notebooks/` — demo notebook
- `tests/` — minimal unit tests
- `.github/workflows/` — CI to run tests
- `requirements.txt` — pinned, minimal dependencies

---

## Reproducibility & Experimental Design 💡

- Use `Config` dataclass and YAMLs to record experimental settings. Save the YAML alongside outputs for traceability.
- Controls: use the seed field to run repeated seeds for variance estimation.
- Baselines and ablations: the simple model and synthetic data make it easy to add and document ablations.

---

## Report (for submission) 📝

This section provides a ready-to-use report template that you can adapt for CA24. It is written as a ready-to-copy `report.md` which includes the required sections for a curriculum assignment: Abstract, Introduction, Methods, Experiments, Results, Reproducibility, and Conclusion.

### report.md

```markdown
# CA24 Report

## Abstract

We present a concise reproducible experiment demonstrating how to map a simple regression problem to an MLP model with configurable components. The repository includes import-safe modules, YAML-based configs, and testing scaffolding to ensure reproducibility and ease of extension.

## Introduction

Outline the problem and the theoretical motivation. For this CA we use a synthetic linear regression task to validate the training pipeline and measurement logging.

## Methods

- Data: synthetic regression with known linear ground truth and additive Gaussian noise.
- Model: MLP with ReLU activations.
- Loss: Mean Squared Error (MSE), implemented via `WeightedMSE` to show how custom loss terms can be included.
- Optimization: Adam with configurable learning rate.

### Mathematical details

Given inputs x in R^d, target y in R, assume y = w^T x + epsilon where epsilon ~ N(0, sigma^2). Model approximates y_hat = f_theta(x). The MSE loss is L = E[(y - y_hat)^2].

## Experiments

1. Baseline experiment with default config (`configs/default.yaml`).
2. Ablation: vary hidden layer sizes and learning rate to demonstrate sensitivity.
3. Repeats: run 5 seeds for each setting and report mean ± std of final validation loss.

## Results

Include figures showing training loss curves across epochs and a table summarizing final losses across hyperparameter settings and seeds.

## Reproducibility

- All code is provided under `src/` and import safe.
- Configs are YAML; record which config file produced which output.
- Use the GitHub Actions CI to verify basic imports and tests; heavy runs should be documented in notebooks and not run in CI.

## Conclusion

Summarize key findings and possible extensions (more complex datasets, additional loss terms, etc.).

## Appendix

Provide exact command lines, seed lists, and data generation details so results can be re-run easily.
```

---

## Notes & Next steps ✨

- Extend `src/model.py` with more architectures for an actual research project.
- Add evaluation scripts, figure generation, and experiment logging backend (e.g., `tensorboard`, `wandb`) as needed.

---

## License

This CA example is released under the project license (see `LICENSE`).














