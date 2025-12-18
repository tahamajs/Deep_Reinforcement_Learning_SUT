# CA27 — Curriculum Assignment 27 ✅

## Overview

This repository contains a compact research-style implementation for meta-learning with two baseline algorithms: Model-Agnostic Meta-Learning (MAML) and Recurrent Meta-RL (RL²). The goal of CA27 is to provide import-safe, tested implementations, and a reproducible experiment scaffold with clear documentation and a deliverable report.

This README contains everything you need to run experiments, reproduce results, and prepare the deliverable report.

---

## Quick features

- Import-safe Python modules under `src/` with type hints and docstrings 🔧
- Notebook for running experiments and visualizations (non-executed, safe) 📓
- Unit tests in `tests/` to validate core pieces ✅
- Reproducible config files under `configs/` (YAML) ⚙️
- A template report `REPORT.md` with suggested structure for the write-up 📝

---

## Installation

Recommended: create a virtual environment and install the requirements.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# optional: dev/test dependencies
python -m pip install pytest
```

Notes:
- The code supports Gym versions with both old and new reset/step APIs.
- PyTorch 1.9+ is required (see `requirements.txt`).

---

## Project layout

- `src/` - main package
  - `config.py` - dataclasses for experiment and algorithm hyperparameters
  - `maml.py` - MAML implementation (policy network, inner/outer loops)
  - `rl2.py` - RL² recurrent policy and trainer
  - `tasks.py` - Task dataclasses and task distribution
  - `utils.py` - trajectory helpers and GAE/returns
- `notebooks/` - example notebook to run experiments and generate figures
- `configs/` - YAML config for experiments (default settings)
- `tests/` - unit tests
- `pictures/`, `results/` - outputs (created by experiments)
- `REPORT.md` - assignment report template and suggestions

---

## Usage (Quickstart)

This repository is intentionally import-safe; the notebook `notebooks/meta_learning_experiment.ipynb` demonstrates how to run experiments and produce figures without modifying module imports.

- To run the notebook: open it in Jupyter and follow the cells. The notebook is non-executed in the repo to keep it reproducible and safe.

- To run tests locally:

```bash
pytest -q
```

---

## Implementation notes

- MAML (`src/maml.py`): simple MLP policy, inner-loop adaptation using manual SGD updates, and Adam-based meta-optimizer for the outer loop. Second-order derivatives are enabled for inner-loop gradients by default.

- RL² (`src/rl2.py`): LSTM-based recurrent policy that encodes past observations, actions and rewards into a hidden state. Includes a PPO-style per-task inner training loop for fast adaptation.

- Utilities (`src/utils.py`): consistent Trajectory container, robust `collect_trajectory` that supports multiple Gym API signatures, and GAE/returns helper functions.

---

## Experiments & reproducibility

A typical experiment flow:
1. Choose an algorithm and set hyperparameters in `configs/default.yaml` or create a YAML file with `ExperimentConfig`.
2. Use the notebook to wire up the experiment and seed the RNG for reproducibility.
3. Save figures to `pictures/` and numeric results to `results/` with descriptive filenames (include seed and timestamp).

Tips:
- Keep experiments small in the notebook when iterating; scale up only for final runs.
- Save random seeds and config files alongside results for reproducibility.

---

## Tests & quality checks ✅

- Tests are located in `tests/` and are run with `pytest`.
- The unit tests cover package initialization, policy forward passes, hidden state shapes, and utility behaviours (including Gym API compatibility).

CI: a minimal GitHub Actions workflow is included at `.github/workflows/ci.yml` to run the test suite on pushes and pull requests.

Makefile: use `make test` to run tests locally or `./scripts/run_tests.sh` for a quick runner.

---

## Report / Deliverable

See `REPORT.md` for a full template: abstract, related work, method descriptions, experimental setup, results, and reproducibility checklist.

---

## Contribution and coding conventions

- Use Python 3.10+, type hints, and dataclasses for configuration.
- Keep modules import-safe (no side effects at import time).
- Preserve theoretical comments and document choices (e.g., number of inner steps, trajectory lengths).

---

## License

This assignment code follows the repository's top-level license. Do not include secrets or private data in commits.

---

If you prefer, I can also prepare a PDF of the report or add a Makefile / script to run experiments in a reproducible manner. Let me know which you'd prefer. ✨













