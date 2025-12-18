# CA28 — Curriculum Assignment 28

## Overview

CA28 is a comprehensive assignment template for research and coursework. This README provides theoretical context, implementation mapping, experimental recommendations, and deliverables in a long-format style.

## Learning Goals

- Build modular, import-safe code for algorithms.
- Map math to code with explicit shape/dtype contracts.
- Run reproducible experiments and save artifacts.
- Produce publication-quality visualizations.

## Expected Files

- `src/` modules.
- `notebooks/` for demos.
- `configs/` for YAML configurations.
- `tests/` for pytest tests.

## Problem Description

Implement Deep Q-Network (DQN) to solve the CartPole environment from OpenAI Gym. The goal is to train an agent that can balance a pole on a cart by applying appropriate forces to the cart.

### Research Question
Can DQN effectively learn a policy to balance the CartPole for at least 195 steps on average over 100 consecutive episodes?

### Experiments
- **Baseline**: Train DQN with the default hyperparameters.
- **Ablations**: Vary learning rate, batch size, and epsilon decay to see their impact.
- **Seed Sweeps**: Run multiple seeds to assess variance in performance.

Implementation notes

- Use dataclasses and YAML loader for configs.
- Provide seeding utilities in `src/utils.py`.
- Keep training loops out of import-time code.

Experiments

- Baseline vs proposed method, ablations, seed sweeps.

Appendix: Padding

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












