# CA31 — Advantage Actor-Critic (A2C) Implementation ✅

## Overview

This assignment implements the Advantage Actor-Critic (A2C) algorithm in PyTorch and includes a small, fully-tested, reproducible bandit example for quick sanity checks. The project is organized for reproducible experiments, unit testing, and clear results reporting.

---

## Quick start 🔧

1. Create a virtual environment and install the requirements:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the unit tests (fast and recommended before experimenting):

```bash
python -m pytest tests/ -v
```

3. Run the small bandit experiment (deterministic with `seed`):

```bash
python -m ca31.train --config configs/experiment.yaml
```

4. Explore the A2C notebook for training and evaluation:

```bash
jupyter notebook notebooks/a2c_training.ipynb
```

---

## Reproducibility & Determinism 🎯

- All experiments use explicit seeding. The helper `set_seed` returns a
  `numpy.random.Generator` for per-run determinism (see `src/ca31/utils.py`).
- Configs are YAML files under `configs/` (e.g. `a2c.yaml`, `experiment.yaml`).
- Results and CSV exports are saved under `results/` when requested by config.

---

## Experiments & Results 📊

Included experiments and expected behavior:

- Bandit example (`src/ca31/`) — deterministic runs, shows eps-greedy learning.
- A2C on CartPole-v1 (not fully executed in CI) — typical results in this repo
  showed mean rewards ≈450 after 50k steps (with the example hyperparameters).

For full experimental details and a short paper-style report, see `REPORT.md`.

---

## Project structure 🔍

- `src/` — Core library: `agent.py`, `model.py`, `config.py`, `utils.py` and a
  small deterministic bandit module under `src/ca31/`.
- `configs/` — Example YAML configs for bandit and A2C experiments.
- `notebooks/` — Notebook for interactive training and visualization.
- `tests/` — Unit tests (determinism, small units, and API checks).
- `results/` — Place to store CSVs and figures produced by runs.

---

## Development notes & style 🔧

- Python 3.10+, typed functions and small, import-safe modules.
- Keep experiments reproducible by passing RNGs explicitly (see `ca31.train`).
- Avoid heavy side effects on import — notebooks and training scripts are entry points.

---

## How to cite / credit 🙏

This repository is a pedagogical implementation used for coursework and research
examples. If you reuse code or results, please cite the project and include a
short note in your README or paper.

---

## Report

A compact report is included as `REPORT.md` (summary, methods, experiments,
results and reproducibility information). See it for exact metrics, plots,
hyperparameter tables and implementation details.

---

If you want, I can also add example figures, CI hooks, or a small `Makefile` to
automate running experiments and generating the report.













