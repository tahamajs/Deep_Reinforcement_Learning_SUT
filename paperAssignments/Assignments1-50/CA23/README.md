# CA23 — Curriculum Assignment 23

## Overview

CA23 is an assignment scaffold intended for research-style projects. This README follows the established long-format CA template and provides objectives, implementation guidance, experiment design, and deliverables suitable for coursework or research prototypes.

## Learning Outcomes

- Implement reproducible experiments and logging.
- Map mathematical derivations precisely to code implementations.
- Produce well-documented, import-safe modules under `src/`.
- Generate publication-quality visualizations saved programmatically.

## Repository Layout

- `src/` — core modules (`config.py`, `model.py`, `losses.py`, `data.py`, `utils.py`).
- `notebooks/` — experiment notebooks and demo runs.
- `configs/` — example YAML configurations for debug and default runs.
- `tests/` — pytest-based unit and smoke tests.

## Problem Statement & Motivation

Select a concrete research question (e.g., uncertainty-aware exploration, robust policy optimization, or a constrained objective). State a clear hypothesis and outline how experiments will validate it.

Math & Objective (example)

- Notation: s (state), a (action), r (reward), \gamma (discount).
- Objective: maximize expected return with optional regularizers or constraints.

Policy-gradient sketch

\nabla*\theta J(\theta) = E*{s,a}[\nabla_\theta \log \pi_\theta(a|s) A(s,a)] - \lambda \nabla*\theta R(\pi*\theta)

Mapping to code

- Implement losses in `src/losses.py` and models in `src/model.py`.
- Add shape and dtype assertions for safety.

Implementation Guidance

- Centralize hyperparameters in `src/config.py` using dataclasses.
- Seed RNGs consistently in `src/utils.py`.
- Keep training loops in notebooks or scripts guarded by main blocks.

Experiments & Evaluation

- Baseline comparison, ablation studies, and seed sweeps are recommended.
- Log metrics to CSV and save figures to `pictures/`.

Visualization & Saving

- Use matplotlib/seaborn for plots and save with dpi=300.

Testing

- Add tests to verify imports, forward passes, and loss computations.

Appendix: Microtasks and padding

1. Micro-task 1
2. Micro-task 2
3. Micro-task 3
4. Micro-task 4
5. Micro-task 5
6. Micro-task 6
7. Micro-task 7
8. Micro-task 8
9. Micro-task 9
10. Micro-task 10
11. Micro-task 11
12. Micro-task 12
13. Micro-task 13
14. Micro-task 14
15. Micro-task 15
16. Micro-task 16
17. Micro-task 17
18. Micro-task 18
19. Micro-task 19
20. Micro-task 20
21. Micro-task 21
22. Micro-task 22
23. Micro-task 23
24. Micro-task 24
25. Micro-task 25
26. Micro-task 26
27. Micro-task 27
28. Micro-task 28
29. Micro-task 29
30. Micro-task 30










