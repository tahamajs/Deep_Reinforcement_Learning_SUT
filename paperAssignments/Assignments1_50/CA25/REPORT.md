# CA25 — Report Template

**Title:** Short descriptive title (max 10 words)

**Authors:** Your Name(s)

**Abstract**
Provide a concise summary (3–5 sentences) covering the problem, method, and short result.

## 1 Introduction
- Brief motivation and context.
- Single-paragraph statement of the hypothesis or question being tested.

## 2 Methods
- **Model:** describe the MLP architecture and hyperparameters used (refer to `src/model.py`). Include layer sizes, activation functions, and initialisation scheme.

**Formulas to include** (copy or adapt into your report):

- MLP forward pass (for L layers):

```tex
h^{(0)} = x \\
z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)} \\
h^{(l)} = \phi(z^{(l)}) \quad \text{for } l=1,\dots,L-1 \\
y = W^{(L)} h^{(L-1)} + b^{(L)}
```

- ReLU: \(\phi(z) = \max(0, z)\)

- MSE loss (regression):

```tex
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
```

- Cross-entropy (classification logits):

```tex
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{z_{i,c_i}}}{\sum_{c} e^{z_{i,c}}}
```

- **Data:** describe the synthetic datasets and how they were generated (`src/data.py`). Include the sampling process and noise model for regression.
- **Loss / training:** describe loss functions and training loop (`src/losses.py`, `train.py`). Include optimizer, lr schedule (if any), batch size, and stopping criteria.


## 3 Experimental Setup
- List the configs used (copy the relevant `configs/*.yaml` block into the report).
- Number of trials / seeds and any hyperparameter sweeps.
- Evaluation metrics.

## 4 Results
- Include plots (for example, `outputs/<run>/pictures/loss.png`).
- Report final metrics and short tables if appropriate.

### Figures
Include the following figures (place files in `outputs/<run>/pictures/`):

- `loss.png` — training and validation loss vs epoch. Caption example: "Training and validation loss across epochs (mean ± std across seeds if available). Config: `configs/example.yaml`."
- `predictions.png` — true vs predicted scatter (regression). Caption example: "True vs predicted on validation set; dashed line is $y=x$. Report RMSE in caption."
- `confusion_matrix.png` — normalized confusion matrix for classification. Caption example: "Normalized confusion matrix (rows=true, cols=predicted)."

LaTeX snippet to include a figure in `rep.tex` or your report:

```tex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.7\textwidth]{outputs/example_run/pictures/loss.png}
  \caption{Training and validation loss across epochs.}
  \label{fig:loss}
\end{figure}
```

## 5 Discussion
- Summarize findings and whether they support the hypothesis.
- Limitations and possible follow-ups.

## 6 Reproducibility & Artifacts
- Command used to run experiments (include config path):

```bash
python -m paperAssignments.Assignments1-50.CA25.train --config configs/example.yaml
```

- Files saved by default: `used_config.yaml`, `model.pt`, and `pictures/loss.png` under the `save_dir`.
- Note any software/hardware specifics (PyTorch version, CPU/GPU).

## Appendix
- Hyperparameter tables or extended plots.

---

Fill the sections above with your experiment details. Keep the report to 1–2 pages for coursework submissions unless otherwise specified by the instructor.
