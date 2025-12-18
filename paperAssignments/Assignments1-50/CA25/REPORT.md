# CA25 — Report Template

**Title:** Short descriptive title (max 10 words)

**Authors:** Your Name(s)

**Abstract**
Provide a concise summary (3–5 sentences) covering the problem, method, and short result.

## 1 Introduction
- Brief motivation and context.
- Single-paragraph statement of the hypothesis or question being tested.

## 2 Methods
- Model: describe the MLP architecture and hyperparameters used (refer to `src/model.py`).
- Data: describe the synthetic datasets and how they were generated (`src/data.py`).
- Loss / training: describe loss functions and training loop (`src/losses.py`, `train.py`).

## 3 Experimental Setup
- List the configs used (copy the relevant `configs/*.yaml` block into the report).
- Number of trials / seeds and any hyperparameter sweeps.
- Evaluation metrics.

## 4 Results
- Include plots (for example, `outputs/<run>/pictures/loss.png`).
- Report final metrics and short tables if appropriate.

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
