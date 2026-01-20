# HW14: Safe Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Safety](https://img.shields.io/badge/Type-Safe--RL-red.svg)](.)

## Overview

This assignment explores Safe Reinforcement Learning: methods that maximize reward while enforcing safety constraints (hard/soft/probabilistic). The folder contains lecture-style notes, implementation notebooks, and solution write-ups.

## Install (quick)

1. Create and activate a venv (recommended):

```bash
python -m venv .venv && source .venv/bin/activate
```

2. Install dependencies for HW14:

```bash
python -m pip install --upgrade pip
python -m pip install -r homeworks/HW14_Safe_RL/requirements.txt
```

Notes: `requirements.txt` contains minimal pins for PyTorch, Gymnasium, NumPy, Matplotlib, imageio and tqdm. Adjust GPU/torch CUDA wheel as needed for your machine.

## Quickstart — run the notebook

1. Open the notebook in Jupyter / Colab:
   - `homeworks/HW14_Safe_RL/code/HW14_Notebook.ipynb`
2. Inspect cells (no cells are executed by default after these edits). The notebook contains implementations for:
   - PPO with Lagrangian multiplier (PPO-Lagrangian)
   - Simplified Constrained Policy Optimization (CPO)
   - A lightweight SafetyLayer (action projection / shield)
3. Run cells interactively to train/evaluate. See `homeworks/HW14_Safe_RL/CODE.md` for example commands and recommended configs.

## File structure

```
HW14_Safe_RL/
├── code/
│   └── HW14_Notebook.ipynb        # Notebook with implementations (unexecuted)
├── answers/
│   └── HW14_Solutions.md         # Detailed written solutions (theory + experiments)
├── reports/
│   └── HW14_Questions.pdf        # Assignment questions / brief
├── Homework-14-Template/         # LaTeX template and assets
├── requirements.txt              # Minimal Python deps for the assignment
└── README.md
```

## Contents & pointers

- Theory and algorithms: the notebook and `answers/HW14_Solutions.md` contain background on CMDPs, CPO, PPO-Lagrangian, CVaR, Control Barrier Functions, and verification approaches.
- Implementations: `HW14_Notebook.ipynb` includes self-contained classes: `PPOLagrangian`, `CPOAgent`, and `SafetyLayer`, plus training/eval helpers (`train_ppo_lagrangian`, `train_cpo`, `evaluate_agent`).
- Template: `Homework-14-Template/main.tex` is provided for report writing (IEEE-like layout assets included).

## Recommended workflow

1. Create venv and install deps (see Install).
2. Open the notebook and run the dependency/cell blocks in order.
3. Use small debug configs first (short episodes, fewer episodes) to verify correctness before long runs.
4. Save evaluation videos and figures to a local `outputs/` folder.

## Citation & references

See `answers/HW14_Solutions.md` for references and suggested readings (Achiam et al. 2017 CPO, García & Fernández 2015 survey, Tamar et al. 2015, etc.).

---

Last updated: 2025-12-18



