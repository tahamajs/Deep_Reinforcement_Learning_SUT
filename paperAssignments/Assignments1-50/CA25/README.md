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

Suggested export commands:

```bash
# Convert markdown report to PDF (requires pandoc and LaTeX)
pandoc REPORT.md -o report.pdf --pdf-engine=xelatex
# Or compile the LaTeX template directly (requires a TeX installation):
pdflatex rep.tex && pdflatex rep.tex
```

Tip: run the included script to create placeholder figures used by the LaTeX report:

```bash
python scripts/make_placeholder_figures.py --out outputs/example_run/pictures
```

## Notes & Best Practices 💡

- Keep notebooks small and exploratory; put any production code in `src/` so it stays import-safe.
- Never commit secrets or large data; include small synthetic examples for unit tests.
- When adding experiments, include a small README snippet explaining the config used and number of seeds.

---

## Mathematical Details & Formulas ✏️

Below are the canonical equations used in the example included in this repo. Include these in your report when discussing methods.

### MLP forward pass
For input x \in R^{B×D} and an L-layer MLP with weights W^{(l)} and biases b^{(l)}, the hidden states are defined by

\begin{align}
    h^{(0)} &= x, \
    z^{(l)} &= W^{(l)} h^{(l-1)} + b^{(l)}, \
    h^{(l)} &= \phi(z^{(l)}) \quad \textrm{for } l = 1,\dots,L-1, \
    y &= W^{(L)} h^{(L-1)} + b^{(L)}
\end{align}

where \phi(·) is the elementwise ReLU (ReLU(z) = max(0, z)). For classification, apply softmax to logits to get class probabilities.

### Loss functions
- Mean squared error (regression):

\begin{equation}
    \mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
\end{equation}

- Cross-entropy (classification logits):

\begin{equation}
    \mathcal{L}_{\mathrm{CE}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{z_{i,c_i}}}{\sum_{c} e^{z_{i,c}}}
\end{equation}

### Evaluation metrics
- For regression: report MSE or RMSE.
- For classification: report accuracy (\frac{1}{N} \sum_i \mathbf{1}[\hat{y}_i = y_i]) and optionally precision/recall.

---

## Figures & Captions (what to save and include) 🖼️

Save clear figure files in `outputs/<run>/pictures/` and reference them in your report. Recommended figures:

- `loss.png` — training and validation loss vs epoch. Caption: "Training and validation loss across epochs (mean ± std across seeds if available)." Include axis labels, legend, and a short caption in your report describing hyperparameters used.
- `predictions.png` — (regression) true vs predicted scatter plot. Caption: "True vs predicted values on validation set; ideal line y=x plotted for reference." Include an R^2 or RMSE summary in the caption.
- `confusion_matrix.png` — (classification) normalized confusion matrix. Caption: "Normalized confusion matrix on validation/test set; rows = true, columns = predicted."

Figure best-practices:

- Export at least 150–300 dpi for raster images (PNG) or use vector formats (PDF/SVG) for plots.
- Label axes clearly with units (if applicable) and include legends where needed.
- Add a brief caption (1–2 sentences) that states what is shown and any key numbers (final metric, RMSE, accuracy).

Include for each figure: file path, short caption, and a sentence interpreting the plot.

---

## Contact & Support

If you use or adapt this template, please keep the import-safe structure and add tests for new functionality. For questions about grading or specific assignment requirements, consult the course staff or your instructor.


---

**Author**: CA25 template (course staff) — updated with report template, mathematical details, and figure guidance















