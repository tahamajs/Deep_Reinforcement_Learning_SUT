# CA22 — Curriculum Assignment 22

## Overview

CA22 is a reproducible research scaffold intended for small research-style projects and mini-experiments. The repository contains import-safe core modules, a small synthetic dataset for debug runs, config examples, and minimal tests to validate building blocks.

## Learning Outcomes

- Implement a reproducible research pipeline and config-driven experiments.
- Translate mathematical derivations into stable, testable code.
- Perform reproducible experiments with logging, checkpoints, and a short written report.
- Produce clear visualizations and a concise findings report suitable for peer review.

---

## Repository Layout

- `src/` — core modules (import-safe): `config.py`, `model.py`, `losses.py`, `data.py`, `utils.py`.
- `notebooks/` — experiment notebooks (non-executed in repo; include `demo.ipynb`).
- `configs/` — YAML configuration sets (`default.yaml`, `debug.yaml`).
- `tests/` — pytest-compatible tests for core modules.
- `reports/` — report templates and submission checklist (added).
- `requirements.txt` — minimal Python packages needed to run tests and demos.
- `outputs/` — for runtime artifacts (not committed).

---

## Quickstart

1. Create a virtual environment and activate it (example):

   python -m venv .venv && source .venv/bin/activate

2. Install minimal requirements:

   pip install -r requirements.txt

3. Run unit tests (from repo root):

   pytest -q paperAssignments/Assignments1-50/CA22/tests

4. For quick debug runs, open `notebooks/demo.ipynb` and follow the notebook instructions (the notebook is designed for reproducible runs using `configs/debug.yaml`).

---

## How to run an experiment (recommended workflow) 🔧

1. Create or edit a config file in `configs/` (copy `default.yaml` → `my_experiment.yaml`) and set seeds and hyperparameters.
2. Use the `ExperimentConfig.from_yaml(path)` helper to load settings programmatically.
3. Seed all libraries with `src.utils.set_seed(cfg.seed)` at the beginning of the run.
4. Save checkpoints with `src.utils.save_checkpoint(...)` and resume with `src.utils.load_checkpoint(...)` when needed.
5. Log hyperparameters, random seeds and the config file used alongside outputs and the final report.

---

## Report & Submission (template included) 📝

A sample report template and a submission checklist are provided in `reports/REPORT_TEMPLATE.md`. Your final submission should include:

- A short report (2–4 pages) following the template.
- Config file(s) used for experiments (YAML).
- Key scripts or a notebook that reproduces the main results.
- Figures and captions saved to `outputs/figures/`.

---

## Evaluation rubric (suggested) ✅

- **Correctness & Reproducibility (40%)**: Code runs without errors, seeds/configs documented, results reproducible.
- **Experiment Design (30%)**: Appropriate baselines, metrics, and evaluation.
- **Clarity & Presentation (20%)**: Clear figures, tables, and readable report.
- **Tests & Code Quality (10%)**: Unit tests, type hints, docstrings.

---

## Reproducibility checklist 📋

- [ ] Include `configs/*.yaml` for main experiments
- [ ] Include random seeds and how they were set
- [ ] Include instructions to reproduce plots and key tables
- [ ] Use `torch.save`/`torch.load` for checkpoints and report the checkpoint format

---

## Contributing & Style guides

- Use Python 3.10+, type hints, dataclasses, and explicit docstrings.
- Keep `src/` import-safe (no side-effects on import).
- Keep changes small and well-scoped; add tests for new utilities.

---

## Contact

If you find issues or want to extend the scaffold, open a PR or contact the course staff.

---

## Changes made in this fork

- Added `reports/REPORT_TEMPLATE.md` with a report template and submission checklist
- Added `reports/EXAMPLE_REPORT.md` showing an example short report
- Added `requirements.txt` and `tests/test_complete.py` with additional unit tests covering config parsing, checkpointing, seed determinism, and Lagrange updates
- Added `notebooks/README.md` with usage guidance for the demo notebook


Feel free to expand the scaffold with training loops and notebooks following the guidelines in the original brief.












