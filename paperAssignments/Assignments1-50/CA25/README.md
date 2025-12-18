# CA25 — Curriculum Assignment 25

## Overview

CA25 is a compact, import-safe scaffold for a reproducible toy experiment (MLP on synthetic data) intended for coursework and brief research prototypes. This repository includes code, configuration files, tests, and a report template to help you run experiments, gather artifacts, and write a concise report.

## Learning Outcomes 🎯

- Implement modular, import-safe Python code with type hints and dataclasses.
- Map equations to code and keep clear tensor shape contracts.
- Run reproducible experiments, log outputs, and save artifacts for reports.
- Produce clear visualizations and a short report summarizing results.

## Project Structure 🔧

- `src/` — core library modules: `config.py`, `model.py`, `data.py`, `losses.py`, `utils.py`.
- `configs/` — YAML config files (example experiments).
- `tests/` — unit tests for imports, forward passes, and loss functions.
- `train.py` — lightweight CLI training entrypoint (import-safe).
- `REPORT.md` — report template for the assignment (fill in and export to PDF for submission).
- `requirements.txt` — python dependencies (note: install PyTorch separately for your platform).
- `outputs/` (generated) — model checkpoints, used config, and `pictures/` with saved figures.

## Quickstart ✅

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Install PyTorch following instructions for your OS/GPU availability:
# https://pytorch.org/get-started/locally/
```

2. Run the example experiment (from repo root):

```bash
python -m paperAssignments.Assignments1-50.CA25.train --config configs/example.yaml
```

This will:
- Load the YAML configuration, set seeds, and choose a device automatically.
- Train for a small number of epochs on synthetic regression data.
- Save `used_config.yaml`, a model checkpoint (`model.pt`), and `pictures/loss.png` under the configured `save_dir`.

3. Run unit tests:

```bash
python -m pytest tests -q
```

## Configuration & Usage

- `configs/example.yaml` contains a complete example. Change `task` to `classification` to switch modes and adjust `hidden_dims`, `lr`, etc.
- The `TrainConfig` dataclass in `src/config.py` documents available fields and default values.

## Developing & Extending 🔧

- Add new datasets and register a loader in `src/data.py`.
- Implement new architectures under `src/model.py` and keep them import-safe.
- Add new loss functions in `src/losses.py` and tests under `tests/`.
- Keep training experiments out of import-time code; use `train.py` or small notebooks for exploratory runs.

## Experiment Checklist (for reproducibility) ✅

- [ ] Use a YAML config and save `used_config.yaml` alongside outputs.
- [ ] Log random seeds and device (done by `train.py`).
- [ ] Save model checkpoints and final metrics programmatically.
- [ ] Save one or more figures to `pictures/` (e.g., `loss.png`).
- [ ] Add a short paragraph in `REPORT.md` describing the experiment and include the plot(s).

## Report & Submission (REPORT.md) 📝

A `REPORT.md` template is included — clone it and fill in the sections (title, abstract, methods, experiments, results, discussion, reproducibility). Export to PDF for submission via `pandoc` or your preferred LaTeX tool.

Suggested export command:

```bash
# Convert markdown report to PDF (requires pandoc and LaTeX/Roadmap)
pandoc REPORT.md -o report.pdf --pdf-engine=xelatex
```

## Notes & Best Practices 💡

- Keep notebooks small and exploratory; put any production code in `src/` so it stays import-safe.
- Never commit secrets or large data; include small synthetic examples for unit tests.
- When adding experiments, include a small README snippet explaining the config used and number of seeds.

---

## Contact & Support

If you use or adapt this template, please keep the import-safe structure and add tests for new functionality. For questions about grading or specific assignment requirements, consult the course staff or your instructor.


---

**Author**: CA25 template (course staff) — updated with report template and reproducibility checklist













