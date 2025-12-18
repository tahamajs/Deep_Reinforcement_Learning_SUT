# CA28 — Curriculum Assignment 28 ✅

## Overview

This project implements a standard Deep Q-Network (DQN) to solve the CartPole (OpenAI Gym) control problem and contains:

- A lightweight, import-safe implementation under `src/`.
- YAML configuration under `configs/`.
- Tests under `tests/` to verify core components.
- A demonstration notebook `notebooks/demo.ipynb` to show usage.
- A LaTeX report template `report.tex` ready for compilation.

This README describes how to set up, run, test, and reproduce experiments and how to adapt the baseline for ablation studies.

---

## Quickstart 🔧

1. Create and activate a Python 3.10+ virtual environment:

   python -m venv .venv && source .venv/bin/activate

2. Install dependencies:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

3. Run the unit tests:

   pytest -q

4. Run a short training (example):

   python -c "from src.config import load_config; from src.train import train_dqn; cfg=load_config('configs/config.yaml'); cfg.num_episodes=10; _=train_dqn(cfg)"

> Note: The repository is import-safe: importing `src` does not run training by default.

---

## Project Structure 🗂️

- `src/` — implementation modules:
  - `config.py` — dataclass `Config` and `load_config` utilities.
  - `model.py` — `QNetwork` architecture.
  - `utils.py` — `set_seed` and `ReplayBuffer`.
  - `train.py` — `DQNAgent` and `train_dqn` loop.
- `configs/config.yaml` — default hyperparameters for experiments.
- `notebooks/demo.ipynb` — quick demo of loading config and running a short training.
- `tests/` — pytest tests for core functions and classes.
- `report.tex` — LaTeX report template with sections for experiments and reproducibility.

---

## Reproducibility & Experiments 📋

- The default config is `configs/config.yaml`. Change hyperparameters there or copy it for ablation studies.
- Seed is set via `Config.seed` and `src.utils.set_seed` which seeds Python, NumPy and PyTorch.
- Use `cfg.num_episodes` to reduce iterations when debugging locally.

Suggested experiments:

1. Baseline: use defaults and train for 500 episodes.
2. Ablations: sweep `learning_rate`, `batch_size`, `epsilon_decay`.
3. Seed sweep: repeat with different `seed` values and report mean/std.

Store experimental results (rewards, checkpoints, figures) under an `outputs/` directory per experiment for reproducibility.

---

## Running the notebook 📓

Open `notebooks/demo.ipynb` in Jupyter or VS Code. If imports fail, launch the notebook from the repo root or set `PYTHONPATH=src` before launching.

---

## Tests ✅

Run `pytest -q`. Tests were designed to be fast and avoid long training. They cover:

- `test_utils.py` — replay buffer and seed helper behaviour.
- `test_model.py` — model forward pass shape checks.
- `test_config.py` — config YAML loading.
- `test_train_agent.py` — basic agent initialization and action selection behaviour.

---

## Report (paper-ready) 📝

The `report.tex` contains a complete write-up with Introduction, Method, Experiments, Results, Discussion and Reproducibility sections. Compile it with a LaTeX tool (pdflatex or latexmk). Figures produced by running experiments should be saved as `figures/training_curve.png` and referenced in the report.

---

## Dependencies (pin / reproduce) 🧾

See `requirements.txt` for the exact minimal environment used for development. Typical packages:

- python>=3.10
- numpy
- torch
- gym
- pyyaml
- matplotlib
- pytest

---

## Contribution & Notes ✍️

- Keep the code import-safe and tests fast.
- Use dataclasses and type hints for clarity.
- If you extend the project (Double DQN / Prioritized Replay / Dueling networks), add small tests and a new config file.

---

## License & Contact

This project uses the repository license in the root. For questions open an issue or contact the author listed in the `report.tex`.













