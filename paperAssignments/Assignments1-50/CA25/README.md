# CA25 — Curriculum Assignment 25

## Overview

CA25 is a comprehensive assignment scaffold for research and coursework tasks. This README provides theoretical framing, implementation guidance, experiment recommendations, and required deliverables in a long-form format.

## Learning Outcomes

- Implement modular, import-safe code.
- Map equations to code with clear tensor shape contracts.
- Run reproducible experiments and save artifacts programmatically.
- Produce visualizations suitable for reports.

## Expected Structure

- `src/` — `config.py`, `model.py`, `data.py`, `losses.py`, `utils.py`.
- `notebooks/` — demo experiments.
- `configs/` — YAML configs.
- `tests/` — unit tests.

Problem description

Choose a focused research problem and draft a hypothesis. Provide math derivations and map to code APIs.

Implementation suggestions

- Use dataclasses and YAML loader for configs.
- Centralize seeding and device utilities.
- Keep training loops out of import-time code.

Experiments & Evaluation

- Baseline comparison, ablation studies, and seed sweeps.
- Log metrics and save figures to `pictures/`.

Testing

- Add tests for imports, forward passes, and loss computations.

Appendix: Tasks and padding

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
8. Task 8
9. Task 9
10. Task 10
11. Task 11
12. Task 12
13. Task 13
14. Task 14
15. Task 15
16. Task 16
17. Task 17
18. Task 18
19. Task 19
20. Task 20

---

## Quickstart ✅

Run the example training script (from repository root):

```bash
python -m paperAssignments.Assignments1-50.CA25.train --config configs/example.yaml
```

This will:
- Load configuration from `configs/example.yaml`.
- Train a small MLP on synthetic data for a few epochs.
- Save artifacts in `outputs/` (model checkpoint, `used_config.yaml`, and `pictures/loss.png`).

Run unit tests with:

```bash
python -m pytest tests -q
```

**Installation** ⚙️

Create a virtual environment and install requirements (PyTorch must be installed separately following your platform instructions):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# install torch separately; example:
# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Tips:
- Edit `configs/example.yaml` to change model size, task type (`regression` or `classification`), and training hyperparameters.
- The code follows import-safe conventions (no heavy work at import time).












