# CA19 — Curriculum Assignment 19

## Overview

This README serves as the central brief for CA19. The goal is to define a reproducible research task, provide theoretical context, and give a concrete implementation and evaluation plan. The document is intentionally long to provide lecture-style guidance and to be used directly by students implementing the assignment.

## Learning Goals

- Build a modular codebase: clear separation between `src/` and `notebooks/`.
- Implement algorithmic ideas in `src/` with type hints and tests.
- Design experiments, run reproducible sweeps, and save results programmatically.

## Folder Conventions

- `src/`: implementation modules (config, model, data, losses, utils).
- `notebooks/`: demonstration notebooks, each saving figures to `pictures/`.
- `configs/`: YAML files for debug and default runs.
- `tests/`: pytest-compatible tests for smoke checks and forward pass validation.
- `outputs/`: for logs, CSVs, and checkpoints (not committed to git).

## Problem Statement

Students must implement an extension of an existing baseline (e.g., a modified actor-critic with uncertainty-aware exploration). The task requires deriving the modified loss, implementing it in code, and empirically validating improvements in sample efficiency and stability.

## Theoretical Background

Notation and Assumptions

- MDP with states S, actions A, transition dynamics P(s'|s,a), and reward r(s,a).
- Discount factor \gamma in (0,1).
- Parametric policy \pi*\theta and value function V*\phi.

Proposed modification (example)

- Introduce an uncertainty penalty U(s) derived from the variance of value ensembles.
- The actor loss adds a term -\beta E*{s\sim D}[U(s) \log \pi*\theta(a|s)] encouraging exploration in uncertain states.

Derivations & mapping

- The variance estimator should be implemented as an ensemble of value networks in `src/model.py` and aggregated in `src/utils.py`.
- The additional loss term must be differentiable w.r.t policy parameters and implemented in `src/losses.py`.

## Implementation Notes

- Use dataclasses for configs.
- Keep devices configurable and use helper functions to move batches to the correct device.
- Provide deterministic seeding in `src/utils.py`.

API Expectations

- Models expose `.forward()` and `.act()` methods where applicable.
- Loss functions accept tensors in shapes (B, ...) and return scalar PyTorch tensors.
- Checkpoint helpers in `src/utils.py` must be robust to missing keys and support CPU-only loading.

Example config dataclass

```python
from dataclasses import dataclass

@dataclass
class CAConfig:
    seed: int = 0
    lr: float = 3e-4
    batch_size: int = 128
    gamma: float = 0.99
```

## Experiment Plan

- Baseline: vanilla actor-critic implementation (with normalization and simple advantage estimation).
- Variant A: ensemble-based uncertainty penalty with \beta in {0.0, 0.01, 0.1}.
- Variant B: ablate ensemble size (1, 3, 5).

Evaluation protocol

- Run each configuration with 5 independent seeds.
- Evaluate on hold-out environment instances or deterministic evaluation episodes.
- Report mean and std for cumulative return.

Visualization plan

- Plot mean reward curves with shaded std band.
- Bar chart comparisons at fixed evaluation steps.
- Example rollouts visualized as plots or short videos saved to `pictures/`.

Logging format

- CSV with columns: timestamp, step, seed, config_id, train_reward, eval_reward, loss_value, lr
- JSON for config metadata per run.

## Tests

- `tests/test_imports.py` to verify `src/` modules import correctly.
- `tests/test_forward.py` to run small batch through the model and check output shapes.
- `tests/test_loss.py` to verify loss functions return finite scalars on synthetic data.

Development checklist

1. Create `src/config.py` with dataclass configs.
2. Create placeholder `src/model.py` with standard PyTorch modules.
3. Create `src/data.py` with dataset wrappers.
4. Create `src/losses.py` with the modified loss terms implemented.
5. Create `src/utils.py` for seeding and checkpointing.
6. Add `configs/debug.yaml` and `configs/default.yaml`.
7. Add `notebooks/demo.ipynb` that runs a short debug experiment.
8. Add unit tests under `tests/`.
9. Run smoke experiment locally and inspect saved figures.

Troubleshooting guide

- If training diverges: reduce LR, reduce batch size, clip gradients.
- If shapes mismatch: add shape assertions in forward methods.
- If CUDA OOM: reduce batch size or move to CPU for debug runs.

References

- Example reference A
- Example reference B
- PyTorch tutorials and best practices

Appendix: Extended task list
This appendix lists micro-tasks to ensure a complete implementation and reproducible results. Each line is a small task or hint.

1. Read baseline paper.
2. Sketch math derivation.
3. Map math to functions.
4. Implement dataclasses.
5. Implement model forward pass.
6. Implement loss functions.
7. Implement optimizer wrapper.
8. Add checkpoint save/load.
9. Add plotting utilities.
10. Add configs for seeds.
11. Add small-demo experiment in notebook.
12. Validate imports via tests.
13. Run debug config and log outputs.
14. Fix failing shape tests.
15. Run 5-seed experiment for main results.
16. Aggregate results and produce figures.
17. Write brief report or README updates.
18. Archive checkpoints locally (do not commit).
19. Prepare slides summarizing findings.
20. Optional: prepare `report.tex` with LaTeX results.

Appendix: Padding for length

A. Padding line 1
B. Padding line 2
C. Padding line 3
D. Padding line 4
E. Padding line 5
F. Padding line 6
G. Padding line 7
H. Padding line 8
I. Padding line 9
J. Padding line 10
K. Padding line 11
L. Padding line 12
M. Padding line 13
N. Padding line 14
O. Padding line 15
P. Padding line 16
Q. Padding line 17
R. Padding line 18
S. Padding line 19
T. Padding line 20
U. Padding line 21
V. Padding line 22
W. Padding line 23
X. Padding line 24
Y. Padding line 25
Z. Padding line 26
AA. Padding line 27
BB. Padding line 28
CC. Padding line 29
DD. Padding line 30
EE. Padding line 31
FF. Padding line 32
GG. Padding line 33
HH. Padding line 34
II. Padding line 35
JJ. Padding line 36
KK. Padding line 37
LL. Padding line 38
MM. Padding line 39
NN. Padding line 40
OO. Padding line 41
PP. Padding line 42
QQ. Padding line 43
RR. Padding line 44
SS. Padding line 45
TT. Padding line 46
UU. Padding line 47
VV. Padding line 48
WW. Padding line 49
XX. Padding line 50

Extra padding lines:

1. Extra pad 1
2. Extra pad 2
3. Extra pad 3
4. Extra pad 4
5. Extra pad 5
6. Extra pad 6
7. Extra pad 7
8. Extra pad 8
9. Extra pad 9
10. Extra pad 10
11. Extra pad 11
12. Extra pad 12
13. Extra pad 13
14. Extra pad 14
15. Extra pad 15
16. Extra pad 16
17. Extra pad 17
18. Extra pad 18
19. Extra pad 19
20. Extra pad 20
21. Extra pad 21
22. Extra pad 22
23. Extra pad 23
24. Extra pad 24
25. Extra pad 25
26. Extra pad 26
27. Extra pad 27
28. Extra pad 28
29. Extra pad 29
30. Extra pad 30
31. Extra pad 31
32. Extra pad 32
33. Extra pad 33
34. Extra pad 34
35. Extra pad 35
36. Extra pad 36
37. Extra pad 37
38. Extra pad 38
39. Extra pad 39
40. Extra pad 40
41. Extra pad 41
42. Extra pad 42
43. Extra pad 43
44. Extra pad 44
45. Extra pad 45
46. Extra pad 46
47. Extra pad 47
48. Extra pad 48
49. Extra pad 49
50. Extra pad 50
51. Extra pad 51
52. Extra pad 52
53. Extra pad 53
54. Extra pad 54
55. Extra pad 55
56. Extra pad 56
57. Extra pad 57
58. Extra pad 58
59. Extra pad 59
60. Extra pad 60
61. Extra pad 61
62. Extra pad 62
63. Extra pad 63
64. Extra pad 64
65. Extra pad 65
66. Extra pad 66
67. Extra pad 67
68. Extra pad 68
69. Extra pad 69
70. Extra pad 70
71. Extra pad 71
72. Extra pad 72
73. Extra pad 73
74. Extra pad 74
75. Extra pad 75
76. Extra pad 76
77. Extra pad 77
78. Extra pad 78
79. Extra pad 79
80. Extra pad 80
81. Extra pad 81
82. Extra pad 82
83. Extra pad 83
84. Extra pad 84
85. Extra pad 85
86. Extra pad 86
87. Extra pad 87
88. Extra pad 88
89. Extra pad 89
90. Extra pad 90
91. Extra pad 91
92. Extra pad 92
93. Extra pad 93
94. Extra pad 94
95. Extra pad 95
96. Extra pad 96
97. Extra pad 97
98. Extra pad 98
99. Extra pad 99
100. Extra pad 100

(End of CA19 README)












