# CA23 — Curriculum Assignment 23 ✅

## Summary

CA23 is a research-style RL assignment scaffold that provides clean, import-safe
modules for experiments (policy/value networks, losses, simple data utilities),
example YAML configs, and unit tests to ensure correctness and reproducibility.
This repository is intended as a minimal, well-documented starting point for
implementing policy-gradient style algorithms and evaluation protocols.

---

## Quickstart — Setup (macOS / Linux)

1. Create and activate a virtual environment:

   python -m venv .venv && source .venv/bin/activate

2. Upgrade pip and install dependencies (recommended):

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

3. (Optional) For development install editable package:

   python -m pip install -e .

4. Run tests with pytest (no GPU required):

   python -m pytest -q

> Note: The test suite uses pytest import-or-skip markers for optional runtime
> dependencies (e.g., `torch`, `yaml`). The tests included are unit-level smoke
> checks — they don't execute full training runs.

---

## Repository layout

- `src/` — core modules:
  - `config.py` — dataclass `ExperimentConfig` (YAML loading/validation)
  - `model.py` — `PolicyNetwork`, `ValueNetwork` (PyTorch)
  - `losses.py` — policy/value/entropy losses
  - `data.py` — episode collection and `discounts`
  - `utils.py` — seeding, device helpers, figure saving
- `configs/` — example config files (`default.yaml`, `debug.yaml`)
- `scripts/` — example training script `train.py` (entry point for experiments)
- `tests/` — pytest tests for imports, shapes, and basic correctness

---

## How to run an experiment (example)

Use the provided `scripts/train.py` which reads a YAML config and runs a
lightweight training loop. Example:

  python scripts/train.py --config configs/debug.yaml

The script accepts configuration via YAML and logs results to `runs/` and
saves plots to `pictures/` by default.

---

## Reproducibility & Reporting

- Seed RNGs via `src.utils.set_seed`.
- Use the `configs/` directory for experiment variants and ensure you
  record the exact config used when reporting.
- Save figures programmatically using `src.utils.save_figure` with
  `dpi=300` for publication-quality output.

---

## Report / Paper

A human-readable report and suggested experimental plan are included in
`REPORT.md`. This file outlines the hypothesis, method, evaluation metrics,
and suggested plots and tables suitable for a short course-style paper.

---

## Contributing

- Keep modules import-safe (no heavy side effects on import).
- Add tests for any new functionality; prefer small, focused tests.
- Document any algorithmic choices in `REPORT.md`.

---

## Licensing & Attribution

This repository is provided for educational use. See `LICENSE` for details.

If you use parts of this scaffold in your work, please cite the course and
maintainers appropriately.

---

If anything is missing for your intended experiments (extra metrics, data
loaders, or plotting utilities), open an issue or create a PR describing the
desired additions.












