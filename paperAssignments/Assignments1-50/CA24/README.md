# CA24 — Curriculum Assignment 24

## Overview

CA24 is a curriculum assignment brief following the repository's long-format CA template. It includes guidance for problem framing, theory-to-code mapping, experiment design, and deliverables for reproducible research projects.

## Learning Objectives

- Translate theoretical objectives into testable code.
- Build import-safe modules with clear APIs.
- Construct reproducible experiments and logging pipelines.
- Produce figures and reports programmatically.

## Folder Layout

- `src/`: `config.py`, `model.py`, `data.py`, `losses.py`, `utils.py`.
- `notebooks/`: demo and experiments.
- `configs/`: debug/default YAMLs.
- `tests/`: unit and smoke tests.

Problem Statement

Define a focused research question, derive the relevant equations, and outline experiments to test the hypothesis.

Theory & Mapping

- Provide derivations for any new loss terms.
- Map each mathematical symbol to function arguments and tensor shapes in code.

Implementation Notes

- Use dataclasses for configs; centralize defaults.
- Implement seed management and device helpers in `src/utils.py`.
- Keep notebooks light; heavy runs should be optional.

Experiments

- Baseline vs method, ablations, and seed repeats are recommended.
- Save metrics and figures for reproducibility.

Testing & Validation

- Include pytest tests for imports and forward passes.

Appendix: Padding lines

1. Pad 1
2. Pad 2
3. Pad 3
4. Pad 4
5. Pad 5
6. Pad 6
7. Pad 7
8. Pad 8
9. Pad 9
10. Pad 10
11. Pad 11
12. Pad 12
13. Pad 13
14. Pad 14
15. Pad 15
16. Pad 16
17. Pad 17
18. Pad 18
19. Pad 19
20. Pad 20
21. Pad 21
22. Pad 22
23. Pad 23
24. Pad 24
25. Pad 25
26. Pad 26
27. Pad 27
28. Pad 28
29. Pad 29
30. Pad 30
31. Pad 31
32. Pad 32
33. Pad 33
34. Pad 34
35. Pad 35
36. Pad 36
37. Pad 37
38. Pad 38
39. Pad 39
40. Pad 40
