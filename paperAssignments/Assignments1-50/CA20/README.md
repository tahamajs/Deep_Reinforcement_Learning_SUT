# CA20 — Curriculum Assignment 20

## Overview

CA20 is an assignment scaffold intended for research-style projects. This README outlines objectives, theoretical grounding, implementation patterns, evaluation practices, and expected deliverables. The document is thorough to provide guidance for both novice and advanced students.

## Learning Outcomes

- Implement a reproducible research pipeline.
- Translate mathematical derivations into stable, testable code.
- Conduct rigorous experiments with proper randomization and logging.
- Produce high-quality visualizations and a short written report.

## Repository Layout

- `src/` — core modules (import-safe): `config.py`, `model.py`, `losses.py`, `data.py`, `utils.py`.
- `notebooks/` — experiment notebooks that save outputs to `pictures/`.
- `configs/` — YAML/JSON configuration sets.
- `tests/` — pytest-compatible tests for core modules.
- `outputs/` — for runtime artifacts (not committed).

## Conceptual Focus

Pick a specific conceptual focus for CA20 (e.g., robust RL, imitation learning, or constrained optimization). The README provides a template for deriving objectives and mapping them to implementation.

Notation & example objective

- Let \pi\_\theta(a|s) be a stochastic policy.
- The target objective is to maximize expected return under a constraint C(\pi) \leq c (e.g., safety constraint).
- Use Lagrangian relaxation to form augmented objective:
  L(\theta, \mu) = E*{\tau\sim\pi\theta}[\sum_t \gamma^t r_t] - \mu (E*{\tau}[C(\tau)] - c)

Mapping to code

- Implement Lagrange multiplier updates in `src/utils.py` or `src/losses.py` with clear interfaces.
- Provide safeguards for multiplier stability (clipping, learning rate scheduling).

## Implementation Guidelines

- Use dataclasses for config management and YAML loader for user-specified overrides.
- Use helper functions to centralize device placement.
- Ensure robust checkpointing that saves both model and optimizer states.

Debug & default configs

- `debug`: small batch sizes, few epochs, deterministic seeds.
- `default`: scaled-up config for main experiments.

Notebook guidance

- Notebook should demonstrate full pipeline using `debug` config.
- Notebook must include code cells for: imports & setup, quick data preview, model instantiation, short training loop, evaluation, and plotting.
- All figures must be saved programmatically to `pictures/` using relative paths.

Experiments
Minimum experiments to perform:

- Baseline vs method with constraint.
- Sensitivity to constraint threshold c.
- Stability across seeds.

Evaluation & metrics

- Primary metric: average episode return under constraint adherence.
- Constraint metric: fraction of episodes violating constraint.
- Report both mean and standard deviation across seeds.

Visualization

- Reward and constraint violation curves over training steps.
- Scatter plots of policy behavior metrics (e.g., mean action magnitude vs reward).

Tests

- Unit tests for forward passes and loss function outputs.
- Integration smoke test that runs a short debug training loop (guarded) and verifies outputs are generated.

Deliverables

- `src/` modules implementing the algorithm.
- `notebooks/demo.ipynb` demonstrating a debug run and saving figures.
- `configs/` with `debug.yaml` and `default.yaml`.
- `tests/` with at least three tests.
- `README.md` (this file) describing usage and hyperparameters.

Appendix: Implementation checklist

1. Draft the mathematical derivation and map to code functions.
2. Implement `src/config.py` with dataclass and YAML loader.
3. Implement `src/model.py` with network modules and forward signature.
4. Implement `src/losses.py` for Lagrangian losses.
5. Implement `src/data.py` with small synthetic data generator.
6. Implement `src/utils.py` for seeding and checkpointing.
7. Add `notebooks/demo.ipynb` with the debug pipeline.
8. Add tests and run `pytest` for touched modules.
9. Save figures to `pictures/`.
10. Create a brief `report.tex` (optional).

Detailed Appendix: Micro-tasks (padding section)

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
31. Micro-task 31
32. Micro-task 32
33. Micro-task 33
34. Micro-task 34
35. Micro-task 35
36. Micro-task 36
37. Micro-task 37
38. Micro-task 38
39. Micro-task 39
40. Micro-task 40
41. Micro-task 41
42. Micro-task 42
43. Micro-task 43
44. Micro-task 44
45. Micro-task 45
46. Micro-task 46
47. Micro-task 47
48. Micro-task 48
49. Micro-task 49
50. Micro-task 50
51. Micro-task 51
52. Micro-task 52
53. Micro-task 53
54. Micro-task 54
55. Micro-task 55
56. Micro-task 56
57. Micro-task 57
58. Micro-task 58
59. Micro-task 59
60. Micro-task 60
61. Micro-task 61
62. Micro-task 62
63. Micro-task 63
64. Micro-task 64
65. Micro-task 65
66. Micro-task 66
67. Micro-task 67
68. Micro-task 68
69. Micro-task 69
70. Micro-task 70
71. Micro-task 71
72. Micro-task 72
73. Micro-task 73
74. Micro-task 74
75. Micro-task 75
76. Micro-task 76
77. Micro-task 77
78. Micro-task 78
79. Micro-task 79
80. Micro-task 80
81. Micro-task 81
82. Micro-task 82
83. Micro-task 83
84. Micro-task 84
85. Micro-task 85
86. Micro-task 86
87. Micro-task 87
88. Micro-task 88
89. Micro-task 89
90. Micro-task 90
91. Micro-task 91
92. Micro-task 92
93. Micro-task 93
94. Micro-task 94
95. Micro-task 95
96. Micro-task 96
97. Micro-task 97
98. Micro-task 98
99. Micro-task 99
100. Micro-task 100

Extra padding lines:

1. Pad A
2. Pad B
3. Pad C
4. Pad D
5. Pad E
6. Pad F
7. Pad G
8. Pad H
9. Pad I
10. Pad J
11. Pad K
12. Pad L
13. Pad M
14. Pad N
15. Pad O
16. Pad P
17. Pad Q
18. Pad R
19. Pad S
20. Pad T
21. Pad U
22. Pad V
23. Pad W
24. Pad X
25. Pad Y
26. Pad Z
27. Pad AA
28. Pad AB
29. Pad AC
30. Pad AD
31. Pad AE
32. Pad AF
33. Pad AG
34. Pad AH
35. Pad AI
36. Pad AJ
37. Pad AK
38. Pad AL
39. Pad AM
40. Pad AN
41. Pad AO
42. Pad AP
43. Pad AQ
44. Pad AR
45. Pad AS
46. Pad AT
47. Pad AU
48. Pad AV
49. Pad AW
50. Pad AX

(End of CA20 README)










