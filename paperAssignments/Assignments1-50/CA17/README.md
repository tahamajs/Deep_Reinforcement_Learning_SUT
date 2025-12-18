# CA17 — Curriculum Assignment 17

## Overview

This README documents CA17. It follows the pattern of other CA folders in this repository and provides a place for the assignment brief, evaluation guidelines, and references. Only documentation is included in this folder by default.

## Objectives

- Present the assignment description and success criteria.
- List required deliverables (code, notebooks, report) and suggested evaluation metrics.
- Provide pointers to datasets and preprocessing steps (if applicable).

## Expected Files

- `README.md` (this file)
- `src/` (optional implementation)
- `notebooks/` (experiment/demo)
- `tests/` (unit tests and smoke tests)

## Getting Started

Ask to scaffold code or a notebook template for CA17 and I will add files consistent with the project's coding standards.

## References

- [Placeholder for papers/resources]

## Notes

Adhere to the repository's CLAUDE.md conventions: import-safe modules, type hints, and deterministic seeding.

## Scaffold provided by assistant

I added a minimal, import-safe implementation to help get started:

- `src/config.py` — dataclass `Config` with default hyperparameters.
- `src/model.py` — `MLPPolicy` (PyTorch) suitable for discrete action spaces.
- `src/losses.py` — simple policy gradient and entropy loss helpers.
- `src/data.py` — episode collection helper (Gym/Gymnasium compatible).
- `src/utils.py` — deterministic seeding and checkpoint helpers.
- `src/train.py` — example training loop guarded by `if __name__ == "__main__":`.
- `tests/test_imports.py` — smoke test for import and forward pass.

Run `python -m pytest paperAssignments/Assignments1-50/CA17/tests` to run the smoke test (ensure the repo root is on PYTHONPATH).









