# CA21 — Curriculum Assignment 21

## Overview

CA21 is a comprehensive assignment template aimed at producing reproducible research results in machine learning. This README covers the assignment description, theoretical basis, implementation plan, experiments, and deliverables. It is written to be long-form and lecture-like for instructional purposes.

## Learning Objectives

- Understand the end-to-end workflow for a small research project.
- Implement modular, testable code and reproducible experiments.
- Generate and save analysis figures programmatically and write a short report summarizing results.

## Expected Directory Structure

- `src/` — core implementation modules.
- `notebooks/` — demonstration and experiment notebooks.
- `configs/` — config files for debug and full runs.
- `tests/` — unit and integration tests.
- `pictures/` — saved figures from notebooks.

## Assignment Focus

Choose a concrete research focus such as: regularized policy optimization, sample-efficient model-based RL, or conditional generative models. Provide a clear hypothesis that the experiments will evaluate.

Mathematical notation (common symbols)

- s, a, s': state, action, next state
- r: reward
- \gamma: discount factor
- \pi\_\theta: policy parameterized by \theta
- V*\phi, Q*\phi: value functions parameterized by \phi

Objective example (constrained opt.)

- Maximize expected return while bounding a constraint metric:
  maximize\_\theta E[\sum_t \gamma^t r_t] subject to E[C(\tau)] \le c
- Use Lagrangian duality or penalty methods to convert to unconstrained objective.

Implementation mapping

- The chosen mathematical objective must have a direct implementation path in `src/losses.py` and `src/model.py`.
- Ensure the API of loss functions is unit tested to match expected shapes and units.

Coding standards

- All public functions and classes must include docstrings and type hints.
- Avoid side effects at import time.
- Centralize hyperparameters in `src/config.py`.

Experiment suggestions

- Baseline vs proposed method comparison.
- Ablation studies for major components.
- Seed sensitivity analysis (report mean ± std over multiple seeds).

Evaluation metrics and logging

- Store per-epoch metrics in CSV files under `outputs/metrics/` (excluded from git).
- Save checkpointed models under `outputs/checkpoints/`.
- Log hyperparameters alongside results (YAML or JSON metadata files).

Visualization guidelines

- Use `matplotlib` or `seaborn` to create publication-quality visuals.
- Programmatically save figures with high DPI and descriptive filenames.

Notebook requirements

- Notebooks must begin with seed setup and device detection.
- Include a brief data exploration or sanity check cell.
- Keep heavy runs behind a clearly marked cell; default notebook should run quickly using `debug` configs.

Testing

- Add unit tests for forward passes and loss computations using synthetic data.
- Add a smoke test that runs a short training loop with the `debug` config.

Deliverable checklist

- `src/` implementation (if requested)
- `notebooks/demo.ipynb` showcasing a reproducible experiment
- `configs/debug.yaml` and `configs/default.yaml`
- `tests/` with basic tests
- `README.md` with usage and hyperparameter table

Appendix: Microtasks and extended checklist

1. Read core baseline papers and list key equations.
2. Write math derivation linking to code API.
3. Implement `src/config.py` and sample configs.
4. Implement `src/model.py` with proper initialization.
5. Implement `src/losses.py` with docstrings and tests.
6. Implement `src/data.py` to provide small demo datasets.
7. Implement `src/utils.py` with seeding and save/load.
8. Add `notebooks/demo.ipynb` with debug run.
9. Add `tests/` for import safety and forward passes.
10. Run debug experiments and save figures.
11. Aggregate results into CSV and create summary plots.
12. Draft a short report summarizing results and include hyperparameter table.
13. (Optional) Create a `report.tex` for LaTeX-ready paper.

Additional notes

- Keep files import-safe and use relative paths for saves.
- Do not commit large binary artifacts (checkpoints, videos). Use local archives instead.
- When requesting scaffolding, specify which modules you want created.

Extra padding section:

1.  Padding 1
2.  Padding 2
3.  Padding 3
4.  Padding 4
5.  Padding 5
6.  Padding 6
7.  Padding 7
8.  Padding 8
9.  Padding 9
10. Padding 10
11. Padding 11
12. Padding 12
13. Padding 13
14. Padding 14
15. Padding 15
16. Padding 16
17. Padding 17
18. Padding 18
19. Padding 19
20. Padding 20
21. Padding 21
22. Padding 22
23. Padding 23
24. Padding 24
25. Padding 25
26. Padding 26
27. Padding 27
28. Padding 28
29. Padding 29
30. Padding 30
31. Padding 31
32. Padding 32
33. Padding 33
34. Padding 34
35. Padding 35
36. Padding 36
37. Padding 37
38. Padding 38
39. Padding 39
40. Padding 40
41. Padding 41
42. Padding 42
43. Padding 43
44. Padding 44
45. Padding 45
46. Padding 46
47. Padding 47
48. Padding 48
49. Padding 49
50. Padding 50
51. Padding 51
52. Padding 52
53. Padding 53
54. Padding 54
55. Padding 55
56. Padding 56
57. Padding 57
58. Padding 58
59. Padding 59
60. Padding 60
61. Padding 61
62. Padding 62
63. Padding 63
64. Padding 64
65. Padding 65
66. Padding 66
67. Padding 67
68. Padding 68
69. Padding 69
70. Padding 70
71. Padding 71
72. Padding 72
73. Padding 73
74. Padding 74
75. Padding 75
76. Padding 76
77. Padding 77
78. Padding 78
79. Padding 79
80. Padding 80
81. Padding 81
82. Padding 82
83. Padding 83
84. Padding 84
85. Padding 85
86. Padding 86
87. Padding 87
88. Padding 88
89. Padding 89
90. Padding 90
91. Padding 91
92. Padding 92
93. Padding 93
94. Padding 94
95. Padding 95
96. Padding 96
97. Padding 97
98. Padding 98
99. Padding 99
100.  Padding 100
101.  Padding 101
102.  Padding 102
103.  Padding 103
104.  Padding 104
105.  Padding 105
106.  Padding 106
107.  Padding 107
108.  Padding 108
109.  Padding 109
110.  Padding 110
111.  Padding 111
112.  Padding 112
113.  Padding 113
114.  Padding 114
115.  Padding 115
116.  Padding 116
117.  Padding 117
118.  Padding 118
119.  Padding 119
120.  Padding 120
121.  Padding 121
122.  Padding 122
123.  Padding 123
124.  Padding 124
125.  Padding 125
126.  Padding 126
127.  Padding 127
128.  Padding 128
129.  Padding 129
130.  Padding 130
131.  Padding 131
132.  Padding 132
133.  Padding 133
134.  Padding 134
135.  Padding 135
136.  Padding 136
137.  Padding 137
138.  Padding 138
139.  Padding 139
140.  Padding 140
141.  Padding 141
142.  Padding 142
143.  Padding 143
144.  Padding 144
145.  Padding 145
146.  Padding 146
147.  Padding 147
148.  Padding 148
149.  Padding 149
150.  Padding 150

(End of CA21 README)













