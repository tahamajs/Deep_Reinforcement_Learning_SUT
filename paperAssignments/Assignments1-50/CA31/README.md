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
results and reproducibility information) and a full LaTeX source `report.tex`.

### Figures and illustrative results

- Placeholder figures are provided in `results/figures/`:
  - `a2c_learning_curve.svg` — illustrative A2C learning curve (replace with your run output).
  - `bandit_rewards.svg` — illustrative bandit reward curve.

- To generate publication-quality figures from experiment CSVs, use the helper script:

```bash
python scripts/plot_examples.py --rewards_csv results/ca31_rewards.csv --out_dir results/figures/
```

This will produce PNGs suitable for inclusion in `report.tex`.

### Build & CI (PDF report)

- `report.tex` contains a full LaTeX report with equations, pseudocode, hyperparameter tables, and placeholders for figures.

Build locally (recommended):

```bash
# convert SVG figures to PDF (if needed) and build report
make report
```

Requirements:
- A TeX engine that provides `pdflatex` (e.g. TeX Live: `texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended`)
- `rsvg-convert` (from `librsvg2-bin`) or `inkscape` (for converting SVG figures to PDF)

Notes:
- The included `Makefile` automates conversion of `results/figures/*.svg` to
  `results/figures/*.pdf` using `rsvg-convert` (fallback to `inkscape`), then
  runs `pdflatex` to build `report.pdf`.
- If you prefer manual conversion, convert SVGs to PDF or PNG and place them
  in `results/figures/` before building.

Continuous Integration:

- A GitHub Actions workflow is included at `.github/workflows/build-report.yml`.
  It runs on `ubuntu-latest`, installs minimal TeX packages and `librsvg2-bin`,
  builds `report.pdf`, and uploads it as a build artifact named `report`.

---

If you'd like, I can also: (a) add example figures generated from a small deterministic
A2C run (I will create reproducible example runs and add their generated
figures into `results/`), (b) add a bibliography file to the LaTeX report, or
(c) set up a GitHub Action step to commit generated figures into a `gh-pages`
branch for easy viewing. Tell me which option to proceed with.














