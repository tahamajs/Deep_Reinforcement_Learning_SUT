# CA26 — Curriculum Assignment 26

## Overview ✅

This repository is a compact, reproducible scaffold for a synthetic regression experiment. It includes an MLP model, loss implementations (MSE and Huber), utilities for deterministic experiments and plotting, a simple training script, and unit tests.

## Highlights

- Small, import-safe Python package under `src/` designed for learning and reproducible experiments.
- Example CLI training script (`scripts/run_experiment.py`) that saves model checkpoints and loss plots.
- Unit tests to ensure correctness and shape checks.
- `REPORT.md` contains a paper-style write-up and reproducibility instructions.

---

## Quickstart ⚡

1. Create and activate a virtual environment (Python 3.10+):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run tests (fast and recommended before experiments):

```bash
python -m pytest -q
```

3. Run the example experiment (writes `model.pt` and a loss plot):

```bash
python -m scripts.run_experiment --config configs/default.yaml --out outputs
```

---

## Project structure 🔧

- `configs/default.yaml` — example experiment config (hyperparameters).
- `scripts/run_experiment.py` — example training loop and CLI.
- `src/` — core code:
  - `config.py` — dataclasses and `load_config` helper.
  - `data.py` — `SyntheticRegressionDataset` and `get_dataloader`.
  - `model.py` — `MLP` and activation helper.
  - `losses.py` — `mse_loss` and `huber_loss` (with shape checks).
  - `utils.py` — seeding, dir helpers, plotting, and `FitResult` dataclass.
- `tests/` — unit tests for components and training pipeline.
- `REPORT.md` — paper-style report and experiment suggestions.

---

## Reproducibility & Running Experiments 🔁

- The code uses deterministic seeds (`src/utils.set_seed`). For real experiments, run multiple seeds and average results.
- Use the CLI for single-run experiments. For sweeps, write a small loop or use hydra/other sweep tools.

Example command:

```bash
python -m scripts.run_experiment --config configs/default.yaml --out outputs/seed-42
```

Expected outputs in `outputs/`:
- `model.pt` — PyTorch state dict
- `loss/loss_curve.png` — training loss per epoch

---

## Tests and CI ✅

- A test suite is included; run with `pytest`.
- Tests added cover losses, model shapes, config loading, utils, and a short run of `fit()`.

---

## How to prepare a report / deliverables 📄

Include the following in your submission:

1. `REPORT.md` (this repo includes an initial draft) with:
   - Abstract, methods, experiments, results, and reproducibility details.
2. `outputs/` directory with example artifacts (`model.pt`, `loss_curve.png`).
3. A small notebook in `notebooks/` summarizing aggregated results (mean ± std across seeds).

---

## Contribution & Notes ✍️

- Keep the repo import-safe: modules should not execute heavy work on import.
- Tests should be fast and deterministic.
- If adding external dependencies, document them in `requirements.txt`.

---

## License

MIT















