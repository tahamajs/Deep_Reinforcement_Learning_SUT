# HW11 Meta-Learning — Reproducibility & Experiments

This document explains how to reproduce experiments and build the report for HW11.

Prerequisites
- Python 3.10+
- A Python virtual environment (recommended): python -m venv .venv && source .venv/bin/activate
- Install Python deps (from workspace root or appropriate requirements file):
  python -m pip install --upgrade pip
  python -m pip install torch numpy matplotlib gym jupyter nbconvert

Optional (LaTeX):
- To compile the PDF you will need a TeX installation (MacOS: `brew install --cask mactex` or install TeX Live)
- Alternatively use `latexmk` if available: `brew install latexmk`

Quick reproduction steps
1. Execute the notebook (runs the experiment code and saves outputs):

```bash
# from repo root
jupyter nbconvert --to notebook --execute homeworks/HW11_Meta_Learning/code/HW11_Meta_Learning_Complete.ipynb \
    --ExecutePreprocessor.timeout=600 --output homeworks/HW11_Meta_Learning/code/executed.ipynb
```

2. Generate HTML preview (optional):

```bash
jupyter nbconvert --to html homeworks/HW11_Meta_Learning/code/executed.ipynb --output homeworks/HW11_Meta_Learning/code/executed.html
```

3. Compile the LaTeX report (requires pdflatex or latexmk):

```bash
cd homeworks/HW11_Meta_Learning/Homework-11-Template
latexmk -pdf -quiet main.tex    # preferred
# or
pdflatex -interaction=nonstopmode main.tex
bibtex main (if using bibliography)
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Notes about experiments
- The notebook contains runnable training functions for MAML, RL² and PEARL with recommended hyperparameters (see `APPENDIX A: HYPERPARAMETERS` in the solution files).
- Use multiple seeds (e.g., 42, 123, 456) to get mean and std for reported metrics.
- Save raw run outputs to `homeworks/HW11_Meta_Learning/results/` (create if missing).

Plotting and figures
- Example plotting snippet (Matplotlib):

```python
import json
import matplotlib.pyplot as plt
with open('homeworks/HW11_Meta_Learning/results/maml_runs.json') as f:
    data = json.load(f)
plt.errorbar(data['steps'], data['mean'], yerr=data['std'])
plt.xlabel('Adaptation steps')
plt.ylabel('Average Reward')
plt.savefig('homeworks/HW11_Meta_Learning/Homework-11-Template/figures/maml_adaptation.png')
```

- Embed figures in LaTeX with `\includegraphics[width=0.8\linewidth]{figures/maml_adaptation.png}`

Tips
- If pdflatex is not installed, `latexmk` is easier (automates multiple pdflatex/bibtex runs).
- For long experiments, consider running on a machine with GPU and use a job scheduler or `tmux`/`screen`.

Contact
- If you run into environment issues, ensure Torch is installed for your platform (see https://pytorch.org) and that `jupyter` and `nbconvert` are in your PATH.

---
