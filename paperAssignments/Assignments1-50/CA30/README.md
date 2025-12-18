# CA30 — Curriculum Assignment 30 ✅

## Overview

This repository is a complete scaffold and example implementation for a research-style assignment (CA30). It contains:

- A small, import-safe Python package under `src/` with a minimal reproducible implementation of a learning experiment (config-driven and tested).
- A notebook template in `notebooks/` for running experiments and generating figures (kept as execution templates — do not run heavy experiments during tests).
- `configs/` with example hyperparameter YAMLs and a `report/` folder containing a full paper draft with instructions and figure placeholders.
- `tests/` and a GitHub Actions CI workflow that runs static checks and lightweight unit tests.

Follow this README to understand the project, reproduce results, and extend it to your research question.

---

## Goals & Scope 🎯

- Provide a reproducible, well-documented example that maps math → code → experiments.
- Keep code import-safe and test-only lightweight components to make CI fast.
- Provide a paper draft that can be edited into a final submission.

---

## Quickstart — What to edit

1. Pick a research question and write it in `report/CA30_paper.md` under "Research Question".
2. Add experiments or algorithms in `src/ca30/` and add config files in `configs/`.
3. Use `notebooks/01_experiment_template.ipynb` as an execution template (document your steps there).
4. Run `pytest -q` locally and edit tests in `tests/` to reflect new functionality.

> Note: This repository is deliberately small. The provided model is minimal (numpy-first) so the package is import-safe even if Torch is unavailable. If you have GPU resources, the notebook shows how to enable full Torch training.

---

## Files & Structure 🔧

- `src/ca30/` — implementation (configs, utils, model skeleton, training loop stub)
- `configs/` — YAML configuration presets for experiments
- `notebooks/` — execution templates for experiment runs and figure generation
- `tests/` — unit tests and small functional checks
- `report/` — paper draft, LaTeX/Markdown ready
- `.github/workflows/ci.yml` — CI that runs tests and linting
- `requirements.txt` — minimal runtime/dev dependencies

---

## Reproducibility checklist ✅

- Use deterministic seeds (see `src/ca30/utils.py`).
- Store experiment configs in `configs/` and save outputs under `results/{experiment_name}/`.
- Keep notebooks for analysis; move production code to `src/`.
- Add deterministic tests for key components (examples in `tests/`).

---

## Development & Tests

Install dependencies (recommended inside a virtualenv):

```
python -m pip install -r requirements.txt
```

Run tests and linters locally (CI runs these same checks):

```
pytest -q
```

---

## How to use the notebook template

- Update `configs/example.yaml` with your hyperparameters.
- Open `notebooks/01_experiment_template.ipynb` and run cells step by step to reproduce an experiment.
- Save figures to `report/figures/` and refer to them in `report/CA30_paper.md`.

---

## Paper / Report instructions 📝

- `report/CA30_paper.md` is a complete draft of a conference-style paper (Abstract, Introduction, Methods, Experiments, Results, Discussion, Reproducibility Appendix).
- Replace placeholders (results, figures, quantitative numbers) with your experimental outcomes.
- Use figure placeholders in `report/figures/` and add final PNG/PDF images there.

---

## Example experiment design (suggestion)

1. Baseline algorithm (provided minimal implementation).
2. One or two variants (ablation on an architectural or training choice).
3. Seed sweeps (3–5 seeds per condition) and a short table of averages + standard errors in the report.

---

## Contributing & Style

- Use dataclasses for configs and type hints (Python 3.10+).
- Keep modules import-safe (no side effects on import).
- Add unit tests for new code and keep tests fast.

---

## License & Citation

This repository is MIT-licensed. Please cite this scaffold when used for course work.

---

## Contact

For questions about this scaffold, edit `AGENTS.md` in the root or open an issue in your fork.

---

Happy researching! ✅













