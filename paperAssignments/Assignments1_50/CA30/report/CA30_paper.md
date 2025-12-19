# Title: A Minimal Reproducible Experiment — CA30

## Authors
Your Name Here

## Abstract
This document is a template paper for CA30. It accompanies the code in `src/` and notebooks in `notebooks/`. Replace placeholders with results from your experiments.

## 1. Introduction
Briefly motivate the research question and prior work. Example: "We study the effect of a small architectural change on a synthetic task."

## 2. Research Question
Clearly state the hypothesis you test in the experiments (e.g., "Adding hidden width improves sample efficiency on the synthetic task").

## 3. Methods
- Model: small MLP (described in `src/ca30/model.py`)
- Data: synthetic binary classification used for quick tests
- Implementation: config-driven, deterministic seeds

Include pseudocode here if needed.

## 4. Experimental Setup
- Datasets / synthetic generation
- Baselines and variants
- Hyperparameters (reference to `configs/example.yaml`)
- Evaluation metrics (accuracy, loss)

## 5. Results (placeholder)
Add tables and figures here.

| Condition | Accuracy mean ± std |
|---|---|
| baseline | 0.00 ± 0.00 |

## 6. Discussion
Interpret results and limitations. Add ablation notes.

## 7. Reproducibility Appendix
- Environments: Python 3.10+, see `requirements.txt`
- How to run: Open `notebooks/01_experiment_template.ipynb` and update config.
- Seed and deterministic details: `src/ca30/utils.py`

## Figures
Add figures to `report/figures/` with names like `fig1.png` and refer to them here.

## References
Add references as required.
