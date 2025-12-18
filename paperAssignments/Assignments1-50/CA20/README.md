# CA20 — Curriculum Assignment 20 ✅

## Overview

CA20 is a compact research-oriented scaffold to explore constrained reinforcement learning via a Lagrangian relaxation approach. It contains import-safe core modules, an example debug notebook, configuration presets, and tests to make development and evaluation reproducible.

This repository is intended as a *self-contained* assignment package: you should be able to run a short debug experiment, generate minimal figures, and produce a short written report reproducing the results.

---

## Quickstart 🔧

Prerequisites

- Python 3.10+
- Recommended: use a virtualenv or conda environment

Install dependencies (minimal):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt  # (see note below)
```

Note: this project keeps dependencies minimal (torch, numpy, pyyaml, matplotlib). Add extras as needed for plotting or LaTeX compilation.

Run a short debug training (no GPU required):

```bash
python -m paperAssignments.Assignments1_50.CA20.src.train
# or programmatically
python -c 'from paperAssignments.Assignments1_50.CA20.src import train, config; train.train(config.Config())'
```

Run tests:

```bash
pytest -q
```

---

## Repository Layout

- `src/` — core modules (import-safe): `config.py`, `model.py`, `losses.py`, `data.py`, `utils.py`, `train.py`.
- `configs/` — YAML sets: `debug.yaml`, `default.yaml`.
- `notebooks/` — `demo.ipynb` showing a minimal debug run and how to save figures to `pictures/`.
- `tests/` — pytest tests for unit and integration checks.
- `outputs/` — runtime artifacts (not committed by default).
- `report.md` / `report.tex` — short reproducible report and LaTeX source.

---

## Design & Implementation Details 🔍

Core ideas:

- Policy: simple MLP Gaussian policy (`src/model.py`) with state-independent diagonal std.
- Value: MLP value function (`src/model.py`) trained via MSE to approximate returns.
- Objective: Lagrangian relaxation of a constraint on the expected constraint value, implemented in `src/losses.py`.
- Multiplier: `LagrangeMultiplier` stateful object (`src/utils.py`) updated by projected gradient ascent.

Robustness & reproducibility features:

- Config dataclass with YAML loader (`src/config.py`)
- Seeding utility (`src/utils.set_seed`) that ensures deterministic RNGs across numpy and torch
- Simple checkpointing (`src/utils.save_checkpoint` / `load_checkpoint`)

---

## Experiments & Reproducibility ✅

Minimum experiments recommended (debug-friendly):

1. Baseline (no constraint) vs Lagrangian-constrained policy
2. Sweep of `constraint_c` (e.g., [0.0, 0.05, 0.1, 0.2])
3. Multiple seeds (e.g., [0, 42, 123]) to report mean and std

Typical commands for reproducing a debug run:

```bash
python -c "from paperAssignments.Assignments1_50.CA20.src import train, config; cfg=config.Config(); cfg.epochs=2; cfg.batch_size=32; train.train(cfg)"
```

Saving figures from the notebook:

- The demo notebook `notebooks/demo.ipynb` includes a cell that runs a short training loop with `debug.yaml` and saves reward/constraint curves to `notebooks/pictures/` (relative paths).

---

## Reporting & Deliverables 📝

Deliverables for CA20 should include:

- Clean `src/` implementation with docstrings and type hints
- `notebooks/demo.ipynb` demonstrating a debug pipeline and saving figures programmatically
- `configs/debug.yaml` and `configs/default.yaml` for quick reproduction and larger experiments
- `tests/` covering unit and integration checks
- `report.md` (and optionally `report.tex`) with method, experiments, quantitative results, and figures

See `report.md` for a complete example of a short reproducible report.

---

## Tests ✅

- Unit tests cover policy forward/sample, losses, and multiplier updates.
- Integration test runs a short debug training loop and ensures a checkpoint is created.
- Added test: checkpoint save/load and config YAML loading are unit-tested.

Run tests:

```bash
pytest -q
```

---

## How to extend

- Replace `SyntheticBanditDataset` with a real environment data loader or sampler
- Plug in richer policy/value networks (CNNs, RNNs) and keep the same training loop
- Add logging (TensorBoard, Weights & Biases) and experiment tracking

---

## Report & Reproducibility Checklist ✔️

- [ ] Short `report.md` describing method, experiments, and results
- [ ] Figures saved to `notebooks/pictures/` with relative paths
- [ ] `requirements.txt` or environment spec provided
- [ ] `pytest` passes locally

---

## License & Contact

This repository is released under the MIT License. Please open issues or PRs for corrections or improvements.

---

**If you use this scaffold for coursework or research, cite this repository in your README or report.**

(End of CA20 README)













