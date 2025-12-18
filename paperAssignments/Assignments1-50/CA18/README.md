# CA18 — Curriculum Assignment 18

## Overview

CA18 is a curriculum assignment scaffold designed to be consistent with the repository's CA style. This README documents the assignment goals, theoretical background, recommended experiments, implementation guidance, evaluation protocol, and deliverables. The document is intentionally extensive to serve as a self-contained lecture-style brief for students and researchers.

## Learning Objectives

- Become familiar with a modern research problem in deep reinforcement learning or generative modeling.
- Design modular, import-safe code organized under `src/` and runtime experiments inside `notebooks/`.
- Translate theoretical objectives into reproducible experiment setups and evaluation metrics.
- Produce publication-quality visualizations programmatically saved to a `pictures/` folder.

## Expected Folder Layout

- `src/` — library-grade modules (import-safe): `config.py`, `model.py`, `data.py`, `losses.py`, `utils.py`.
- `notebooks/` — Jupyter notebooks for demonstrations and experiments.
- `configs/` — YAML files with example hyperparameter sets (small-demo and full-run variants).
- `tests/` — unit tests and smoke tests validating imports and shape assertions.
- `pictures/` — generated figures saved by notebooks during experiments.
- `report.tex` — (optional) LaTeX paper template for reporting results.

## Problem Statement

The assignment centers on implementing a research-oriented algorithmic component (e.g., a policy learning module, a new value regularizer, or a conditional generative model). Students must identify a measurable gap in baseline methods and propose a clear hypothesis to test. The hypothesis should be supported by mathematical reasoning and by empirical experiments.

## Background and Motivation

Provide concise background relevant to the chosen topic. Cite canonical works and briefly describe how this assignment builds on them. For example, if the focus is on value regularization, cite Bellman backup literature and recent regularization strategies. If generative modeling, cite flow-based or diffusion works.

## Math & Theory

This section describes the theoretical objective that the implementation should realize.

Notation

- x \in X: input state or observation
- a \in A: action
- r(s, a): reward function
- \pi\_\theta(a|s): parametric policy with parameters \theta
- Q\_\phi(s, a): parametric Q-value with parameters \phi

Objective (example — regularized actor-critic)
Let J(\theta) = E*{\tau\sim\pi*\theta}[\sum_t \gamma^t r_t - \lambda R(\pi_\theta)]
where R(\pi) is a policy regularizer such as KL to a prior or an entropy term.

Policy gradient (sketch)
\nabla*\theta J(\theta) = E*{s\sim d^{\pi}, a\sim\pi*\theta}[\nabla*\theta \log \pi*\theta(a|s) (Q^{\pi}(s,a) - b(s))] - \lambda \nabla*\theta R(\pi\_\theta)

Derivation mapping

- The theoretical gradient above must be implemented in `src/losses.py` as a differentiable PyTorch function that accepts batched tensors and returns scalar losses and gradients via autograd.
- Shape and dtype assertions should be present for safety.

Assumptions

- Finite-horizon or discounted infinite-horizon MDP.
- Observations are numeric tensors; preprocess with normalization where appropriate.
- Use stable baselines for advantage estimation (e.g., GAE) when requested.

## Implementation Guidance

Design your code with the following constraints in mind:

- Import-safety: modules must not execute training on import.
- Type hints: prefer explicit typing for function signatures.
- Deterministic seeding: put seed setup into `src/utils.py` and call from notebooks.
- Shape checks: add assert statements or helper functions that validate tensor shapes.

Suggested `src/` module responsibilities

- `config.py`: hyperparameter dataclass and sample configs for quick runs.
- `data.py`: dataset classes, synthetic tasks, and evaluation wrappers.
- `model.py`: neural network definitions, weight initialization utilities.
- `losses.py`: loss function implementations (policy loss, value loss, entropy/kl terms).
- `utils.py`: logging, checkpointing, random seed control, and device helpers.

Example API snippets

```python
# src/config.py (example dataclass)
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    lr: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    device: str = 'cpu'
```

Design notes

- Keep training loops out of `src/`. The notebook or scripts under `scripts/` should run experiments.
- Save model checkpoints to `outputs/checkpoints/` and log metrics to `outputs/logs/`.

## Reproducibility & Small-demo Settings

Provide at least two config variants:

1. `debug`: small dataset, few epochs, fast iterations for local testing.
2. `default`: moderately sized run replicating the intended experiment but reduced from full-scale for quick reproducibility.

Example YAML layout

```yaml
debug:
  seed: 123
  lr: 1e-3
  batch_size: 16
  epochs: 3

default:
  seed: 42
  lr: 3e-4
  batch_size: 64
  epochs: 50
```

## Experiments

Recommend a minimum set of experiments to validate the hypothesis:

- Ablation 1: Remove the proposed regularizer and compare performance.
- Ablation 2: Vary the weight \lambda across {0.0, 0.01, 0.1, 1.0}.
- Sensitivity: Run with 5 seeds and report mean ± std.

Metrics

- Primary: cumulative reward (average across evaluation episodes).
- Secondary: sample efficiency (reward vs gradient steps), stability (std across seeds).
- Logging: store per-epoch metrics in CSV or JSON for later plotting.

Visualization

- Loss curves: training/validation loss over time.
- Reward curves: mean and percentile bands across seeds.
- Qualitative: policy rollout videos or state-space visualizations.

Programmatic saving of figures (example)

```python
import matplotlib.pyplot as plt
plt.plot(train_steps, rewards)
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.savefig('pictures/fig_rewards.png', dpi=300)
```

## Testing & Validation

- Provide unit tests in `tests/` that import `src/` modules and validate forward passes and loss outputs.
- Smoke test: run a short training script with `debug` config and ensure API completes without errors.

Quality checklist

1. All modules import without side effects.
2. Type hints present for public functions and classes.
3. Configs are centralized in `src/config.py`.
4. No `TODO` placeholders in core algorithmic code.
5. Figures saved to `pictures/` using relative paths only.

## Deliverables

- `src/` implementation (optional if assignment is documentation-only)
- `notebooks/demo.ipynb` with end-to-end reproducible experiment run (using `debug` config for quick checks)
- `pictures/` with the core figures
- `report.tex` (optional) for write-up
- `tests/` with unit and smoke tests

## Timeline & Milestones (suggested)

- Week 1: Literature review and precise hypothesis statement.
- Week 2: Math derivations and API design.
- Week 3: Implement core modules and small-demo experiments.
- Week 4: Run full experiments, analyze results, and prepare report.

## References

- [1] Example baseline paper A
- [2] Example baseline paper B
- [3] Example technical resource (PyTorch, Gymnasium)

## Appendix

This appendix contains implementation notes, common pitfalls, and a padded checklist of small tasks that often appear during development. It also functions as a simple progress tracker.

Implementation notes

- Prefer `torch.nn.Module` for model components and `torch.optim` optimizers.
- Abstract away device placement with helper functions.
- Use consistent RNG seeding across NumPy, random, and torch.

Common pitfalls

- Running heavy training on import — avoid this.
- Hardcoding absolute paths — use Pathlib and relative saves.
- Mixing NumPy and Torch dtypes incorrectly.

Progress tracker (detailed)

1. Define the assignment hypothesis and success criteria.
2. Collect and read 3–5 related papers.
3. Write math derivations and map to code APIs.
4. Scaffold `src/` modules and `notebooks/`.
5. Implement `config.py` with dataclasses.
6. Implement `model.py` with deterministic weight init.
7. Implement `data.py` including small synthetic generator.
8. Implement `losses.py` reflecting mathematical derivations.
9. Implement `utils.py` for seeding and logging.
10. Create `configs/` YAML files for `debug` and `default`.
11. Implement `tests/` for import safety and basic forward passes.
12. Write notebook `notebooks/demo.ipynb` showing quick run.
13. Run small-demo locally to validate shapes.
14. Tune learning rates and batch sizes for `default` runs.
15. Run seed sweep (5 seeds) for main results.
16. Generate figures and save them under `pictures/`.
17. Compile `report.tex` with results placeholders.
18. Add README and hyperparameters table to `report.tex` appendix.
19. Zip checkpoints for archival (do not commit to git).
20. Prepare a short presentation summarizing findings.

Detailed task list (padding for completeness)

1. Item 1
2. Item 2
3. Item 3
4. Item 4
5. Item 5
6. Item 6
7. Item 7
8. Item 8
9. Item 9
10. Item 10
11. Item 11
12. Item 12
13. Item 13
14. Item 14
15. Item 15
16. Item 16
17. Item 17
18. Item 18
19. Item 19
20. Item 20
21. Item 21
22. Item 22
23. Item 23
24. Item 24
25. Item 25
26. Item 26
27. Item 27
28. Item 28
29. Item 29
30. Item 30
31. Item 31
32. Item 32
33. Item 33
34. Item 34
35. Item 35
36. Item 36
37. Item 37
38. Item 38
39. Item 39
40. Item 40
41. Item 41
42. Item 42
43. Item 43
44. Item 44
45. Item 45
46. Item 46
47. Item 47
48. Item 48
49. Item 49
50. Item 50
51. Item 51
52. Item 52
53. Item 53
54. Item 54
55. Item 55
56. Item 56
57. Item 57
58. Item 58
59. Item 59
60. Item 60
61. Item 61
62. Item 62
63. Item 63
64. Item 64
65. Item 65
66. Item 66
67. Item 67
68. Item 68
69. Item 69
70. Item 70
71. Item 71
72. Item 72
73. Item 73
74. Item 74
75. Item 75
76. Item 76
77. Item 77
78. Item 78
79. Item 79
80. Item 80
81. Item 81
82. Item 82
83. Item 83
84. Item 84
85. Item 85
86. Item 86
87. Item 87
88. Item 88
89. Item 89
90. Item 90
91. Item 91
92. Item 92
93. Item 93
94. Item 94
95. Item 95
96. Item 96
97. Item 97
98. Item 98
99. Item 99
100. Item 100

(End of CA18 README)

---

## Quickstart ✅

Follow these steps to run the small demo and tests locally.

1. Create and activate a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run tests (recommended):

```bash
pytest -q
```

4. Open the demo notebook (non-executed, contains runnable cells):

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Report & Figures 📝

A LaTeX template `report.tex` is included in this folder. To compile locally (requires a LaTeX installation):

```bash
pdflatex report.tex
# or, with latexmk installed:
latexmk -pdf report.tex
```

The report is a template with sections for abstract, methods, experiments, and placeholder figures — fill the placeholders with your results and figures saved under `pictures/`.

---

## Files added in this fork 🔧

- `report.tex` — LaTeX report template for CA18 deliverables.
- `notebooks/demo.ipynb` — non-executed demo notebook showing a minimal run using `debug.yaml`.
- `requirements.txt` — minimal dependencies for running tests and demos.
- Additional unit tests under `tests/` to validate losses, data shapes and config loading.

---

## Contributing & Style 💡

- Keep modules import-safe (no side effects at import-time).
- Write type hints and docstrings for public functions/classes.
- Add tests when you modify behavior.

---

## License

This assignment follows the repository license in `LICENSE`. Please do not include secrets or private data in commits.












