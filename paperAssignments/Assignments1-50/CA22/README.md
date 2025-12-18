# CA22 — Curriculum Assignment 22

## Overview

CA22 is an assignment scaffold intended for research-style projects. This README outlines objectives, theoretical grounding, implementation patterns, evaluation practices, and expected deliverables. The document mirrors the long-format style used across other CAs and is suitable as a lecture-style brief.

## Learning Outcomes

- Implement a reproducible research pipeline.
- Translate mathematical derivations into stable, testable code.
- Conduct rigorous experiments with proper randomization and logging.
- Produce high-quality visualizations and a short written report.

## Repository Layout

- `src/` — core modules (import-safe): `config.py`, `model.py`, `losses.py`, `data.py`, `utils.py`.
- `notebooks/` — experiment notebooks that save outputs to `pictures/`.
- `configs/` — YAML/JSON configuration sets.
- `tests/` — pytest-compatible tests for core modules.
- `outputs/` — for runtime artifacts (not committed).

## Quickstart (debug)

1. Install minimal requirements: `pip install -r requirements.txt` (create virtualenv first).
2. Run unit tests: `pytest -q paperAssignments/Assignments1-50/CA22/tests`
3. Use `notebooks/demo.ipynb` for an end-to-end debug run (not executed in repo).

## Implementation checklist

Files included in this assignment scaffold:

- `src/config.py` – dataclass `ExperimentConfig` with YAML loader.
- `src/model.py` – `PolicyNet`, `ValueNet`, and `sample_action` helper.
- `src/losses.py` – policy/value and Lagrangian loss composition.
- `src/data.py` – `SyntheticDataset` and `synthetic_episode` generator.
- `src/utils.py` – seeding, checkpointing, and Lagrange updates.
- `configs/debug.yaml` and `configs/default.yaml` – example configs.
- `tests/test_smoke.py` – quick smoke tests for import and forward pass.

Feel free to expand the scaffold with training loops and notebooks following the guidelines in the original brief.











