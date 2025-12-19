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

1. Create and activate a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install -r paperAssignments/Assignments1-50/CA17/requirements.txt
```

3. Run training (from repo root):

```bash
python -m paperAssignments.Assignments1_50.CA17.src.train
```

4. Run tests:

```bash
python -m pytest paperAssignments/Assignments1-50/CA17/tests
```

5. Quick notebook: open `paperAssignments/Assignments1-50/CA17/notebooks/CA17_experiment_template.ipynb` and reduce `total_timesteps` for quick experiments.




## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
- Sutton, R., & Barto, A. (2018). Reinforcement Learning: An Introduction.
- Schulman, J., et al. (2015). High-dimensional continuous control using generalized advantage estimation.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms.

For full citations used in the LaTeX report see `REPORT.bib`.

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













