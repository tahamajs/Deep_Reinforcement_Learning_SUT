# CA26 — Project Report

## Abstract ✅

This project implements a small, reproducible synthetic regression benchmark for investigating regression losses and small MLP architectures. It provides a minimal, import-safe codebase with: a synthetic dataset generator, model and loss implementations, utilities for deterministic runs and plotting, unit tests and a reproducible training script.

---

## 1. Introduction

CA26 is a teaching-focused scaffold intended to: 1) make correctness and reproducibility easy with small, tested modules; and 2) provide a template for short experiments (multi-seed, lightweight sweeps) suitable for homework or reproducible research demos.

Key goals:
- Keep modules import-safe and well-tested.
- Provide a clear mapping from math → code → experiment logs/artifacts.
- Make it easy to reproduce and extend baseline experiments.

---

## 2. Dataset (detailed) 🔬

**Generation**
- Synthetic regression: x ∼ Uniform[0, 1], y = sin(2πx) + ε where ε ∼ N(0, σ²).
- Implemented in `src/data.py` as `SyntheticRegressionDataset`.

**Configurable parameters**
- `n_samples` (default 1000)
- `noise_std` (default 0.1)
- `seed` for deterministic generation

**Recommended usage**
- For experiments, split dataset into train/validation/test folds (simple random split); the code currently returns a dataset object and a `DataLoader` — perform splitting in your experiment notebook or wrap the dataset.
- For reproducibility, run multiple seeds and aggregate metrics.

---

## 3. Models & Initialization 🔧

**Model family**
- `MLP` in `src/model.py`: fully-connected layers with configurable `hidden_dims` and a final `output_dim`.
- Activations supported: `relu`, `tanh`, `identity`.

**Example configuration (from `configs/default.yaml`)**
- `model.input_dim: 1`
- `model.hidden_dims: [64, 64]`
- `model.output_dim: 1`
- `model.activation: relu`

**Notes**
- We use the default PyTorch linear initialization (Kaiming-style for ReLU by default in PyTorch layers). If precise initialization is needed, add explicit initializers in `MLP.__init__`.

---

## 4. Losses (math + code) 📐

- Mean Squared Error (MSE):

  MSE(y, \hat y) = (1/N) ∑_i (y_i - \hat y_i)^2

  Implemented as `mse_loss(pred, target)` in `src/losses.py` with a shape check.

- Huber loss (smooth L1):

  H_δ(r) = 0.5 r^2  if |r| <= δ,  δ(|r| - 0.5 δ) otherwise.

  Implemented as `huber_loss(pred, target, delta=1.0)`; this is useful for robustness to outliers.

**Practical guidance**
- For low-noise synthetic data MSE and Huber with large δ behave similarly.
- Use Huber for datasets with occasional large outliers.

---

## 5. Training procedure & hyperparameters 🏋️‍♀️

**Training loop**
- Provided in `scripts/run_experiment.py` as `fit(cfg, out_dir)`.
- Uses Adam optimizer, callable from CLI via `main()`.

**Default training hyperparameters (`configs/default.yaml`)**
- seed: 42
- batch_size: 128
- lr: 1e-3
- epochs: 30
- device: cpu

**Artifacts produced**
- `model.pt`: saved `state_dict` of the trained model.
- `loss/loss_curve.png`: PNG of the training loss curve (one file per run).

**Good practices**
- Run multiple seeds and aggregate the final metric (mean ± std).
- For CI or quick smoke tests, limit epochs to 1–3.

---

## 6. Evaluation Metrics & Plots 📊

Recommended metrics:
- MSE (mean squared error)
- RMSE = sqrt(MSE)
- MAE (mean absolute error)
- R^2 (coefficient of determination)

Recommended figures to include in a report:
- **Loss curve:** training loss vs epoch (`loss_curve.png`).
- **Predictions vs truth:** scatter plot of y_true vs y_pred on validation/test split.
- **Residual histogram:** distribution of y_true - y_pred (to inspect bias/outliers).
- **Aggregate plot:** mean ± std loss across seeds.

Include numerical summaries (table of mean ± std) for the final epoch metric across seeds.

---

## 7. Experiments & Protocols 🔬

**Baseline experiment**
- Use `configs/default.yaml`, run `fit` for several seeds (e.g., seeds [0, 1, 2, 3, 4]).
- Record final loss and generate plots.

**Ablations**
- Loss comparison: MSE vs Huber (vary δ)
- Capacity comparison: hidden sizes (e.g., (16,), (64,64), (128,128))
- Noise sensitivity: vary `noise_std` in dataset

**Reporting**
- Always show mean ± std for final loss across seeds.
- Provide at least one representative predictions-vs-truth figure and an aggregated loss curve.

---

## 8. Implementation mapping (detailed) 🗺️

- `src/data.py` — dataset generator and `get_dataloader` (shuffling enabled).
- `src/model.py` — `MLP` class and `get_activation` helper.
- `src/losses.py` — `mse_loss`, `huber_loss` with shape checking and clear semantics.
- `src/utils.py` — `set_seed`, `ensure_dir`, `save_loss_curve`, and `FitResult` dataclass. Uses `matplotlib.use('Agg')` for headless plotting.
- `scripts/run_experiment.py` — `fit()` function and CLI wrapper.
- `configs/default.yaml` — default experiment configuration.
- `tests/` — automated tests that validate shapes, loss behavior, config loading, utils, and a short run of `fit()`.

---

## 9. Tests & Continuous Integration ✅

**Test coverage (key tests)**
- `tests/test_losses.py` — sanity checks for `mse_loss` and `huber_loss`.
- `tests/test_model_shapes.py` — ensures `MLP` forward shape.
- `tests/test_config.py` — verifies the YAML -> dataclass loader.
- `tests/test_utils.py` — seeds and `save_loss_curve` behavior.
- `tests/test_train.py` — short smoke test for `fit()` that verifies artifacts are written.

**CI**
- `.github/workflows/ci.yml` runs `pytest` on push/PR (Python 3.10). The plotting backend is set to `Agg` to avoid display errors in headless CI environments.

---

## 10. Reproducibility checklist ✅

Before sharing results, ensure:
- [ ] `requirements.txt` lists all packages and versions used.
- [ ] Each experiment run uses a fixed seed; keep a list of seeds and a short config file per run.
- [ ] Save artifacts: `model.pt`, `loss/loss_curve.png`, and aggregated CSV of final metrics across seeds.
- [ ] Add a short notebook under `notebooks/` that loads artifacts and generates final plots/tables.

Example command sequence:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.run_experiment --config configs/default.yaml --out outputs/seed-42
```

---

## 11. Deliverables (for submission) 📦

Include the following in your deliverable bundle:
- `REPORT.md` (this document, updated with final numbers and figures) ✅
- `outputs/` directory containing example artifacts (`model.pt`, `loss/loss_curve.png`, aggregated CSV of runs) ✅
- Notebook(s) in `notebooks/` reproducing the main figures and aggregated analysis ✅
- Tests passing (`pytest -q`) and CI green ✅

---

## 12. Extensions & Future Work ✨

- Add uncertainty estimates (e.g., MC Dropout, ensembles) and evaluate calibration.
- Add parameter sweeps (hydra or a small sweep runner) and a reproducible logging backend (e.g., CSV, MLFlow).
- Add synthetic tasks with different functions (polynomial, piecewise) to stress test losses.

---

## Appendix: Example aggregation snippet

```python
import json
import numpy as np
import pandas as pd

# Given a directory with several outputs/seed-*/model.pt and loss curves
# Collect final losses (pseudo-code)
rows = []
for seed in seeds:
    # load summary.json or compute from loss file
    rows.append({"seed":seed, "final_loss": final_loss})

df = pd.DataFrame(rows)
df.to_csv("outputs/aggregate.csv", index=False)
print(df.mean(), df.std())
```

---

## 13. Experiment logging & aggregation (recommended) 🧾

To make experiments reproducible and easy to aggregate, adopt a small, consistent run-logging format for every run. Example layout for a run named `outputs/seed-42`:

- `outputs/seed-<n>/metadata.json` — contains run metadata: config, seed, timestamps, final metrics.
- `outputs/seed-<n>/model.pt` — saved model state dict.
- `outputs/seed-<n>/loss/loss_curve.png` — training loss curve.

Example `metadata.json` (recommended keys):

```json
{
  "seed": 42,
  "config": {
    "model": {"input_dim": 1, "hidden_dims": [64,64], "output_dim": 1},
    "train": {"lr": 0.001, "batch_size": 128, "epochs": 30}
  },
  "final_loss": 0.012345,
  "timestamp": "2025-12-19T12:34:56Z"
}
```

Aggregation script (example, save as `scripts/aggregate_results.py`):

```python
from pathlib import Path
import json
import pandas as pd

ROOT = Path("outputs")
rows = []
for d in ROOT.glob("seed-*"):
    m = d / "metadata.json"
    if not m.exists():
        continue
    meta = json.loads(m.read_text())
    rows.append({
        "seed": meta.get("seed"),
        "final_loss": meta.get("final_loss"),
        **meta.get("config", {})
    })

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "aggregate.csv", index=False)
    print(df.describe())
else:
    print("No runs found")
```

Run multiple seeds (example shell snippet):

```bash
for s in 0 1 2; do
  python -m scripts.run_experiment --config configs/default.yaml --out outputs/seed-$s --seed $s
done
python scripts/aggregate_results.py
```

Note: `scripts/run_experiment` can be extended to write `metadata.json` automatically at the end of each run (recommended).

---

## 14. Expected deliverable layout (for submissions)

When packaging results for review, include these files and paths in the archive:

- `README.md` — quickstart and explanation of the experiment
- `REPORT.md` — this document (with final numbers and figures)
- `configs/default.yaml` and any config variants used
- `scripts/run_experiment.py` (or wrapper scripts)
- `scripts/aggregate_results.py` (optional helper)
- `notebooks/` — reproducible notebooks used to generate figures
- `outputs/` — example artifacts (`seed-*/model.pt`, `seed-*/loss/loss_curve.png`, `aggregate.csv`)
- `tests/` — unit tests and guidance to run them

---

## 15. How to reproduce exact numbers (step-by-step)

1. Create a fresh virtual environment and install pinned requirements.
2. Checkout the repository at the commit used for experiments (record the commit SHA in `metadata.json`).
3. Run the same sequence of `scripts/run_experiment` commands with the same seeds and configs.
4. Use `scripts/aggregate_results.py` to collect results and compute mean ± std.
5. Recreate figures from the notebook using the saved artifacts.

Tips:
- Pin package versions in `requirements.txt` for exact reproducibility.
- Record random seeds and the commit SHA in `metadata.json`.

---

## 16. Final checklist before submission ✅

- [ ] All tests pass locally (`pytest -q`).
- [ ] CI (GitHub Actions) passes on the main branch.
- [ ] `REPORT.md` contains final tables and figures with captions.
- [ ] `outputs/` contains representative runs and `aggregate.csv`.
- [ ] Notebooks reproduce the figures in the report without heavy compute.
- [ ] All code changes are committed with descriptive messages.

---

## Credits & License

Author: CA26 scaffold
License: MIT

---

If you'd like, I can:
- Add an actual `scripts/aggregate_results.py` helper file and a small `notebooks/ca26_experiment.ipynb` that runs 3 seeds (non-executed) and produces an aggregated figure and table, or
- Extend `scripts/run_experiment.py` to write `metadata.json` automatically at the end of each run.

Tell me which you'd like (aggregate helper, notebook, instrumenting runs), and I'll add it.

