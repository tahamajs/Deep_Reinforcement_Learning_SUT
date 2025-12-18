CA15 — Report
==============

Authors: Curriculum Assignment 15 (template)

Abstract
--------
This short report documents the minimal algorithmic components, experimental setup, and reproducibility instructions for CA15. The included implementation provides a small MLP policy and value networks, a synthetic dataset for quick checks, a training script, and a set of utilities to save/load checkpoints and manage randomness.

1. Introduction
---------------
The goal of CA15 is to provide a compact, well-documented implementation suitable for educational and research-first use. The method used here is intentionally simple: a policy network trained with a policy-gradient style loss and a value network trained with mean-squared error on synthetic targets. This assignment focuses on reproducibility, clarity, and clean, import-safe API design.

2. Methods
----------
- Model: MLP policy producing action logits and a separate MLP value function.
- Losses: mean-squared error for value regression and a simple REINFORCE-style policy loss (-log p * advantage).
- Data: synthetic dataset with random features, random integer action targets and scalar value targets for quick experimentation.

3. Experimental Plan
--------------------
Although this repository does not include large-scale experiments, the intended small experiments are:

- Vary seed and compare final losses to ensure determinism.
- Vary hidden sizes and learning rates to observe stability of training.
- Use the minimal training loop in `src/train.py` to perform short CPU runs (e.g., epochs=5) and collect `policy_loss`, `value_loss`, and run time.

4. Expected Results and Evaluation
---------------------------------
- The training loop should complete without errors on CPU.
- Because labels are synthetic and random, we do not expect meaningful task performance; instead we evaluate stability (loss decreases moderately) and deterministic behavior across seeding.

5. Reproducibility
------------------
To reproduce results:

- Create a venv and install dev dependencies (PyTorch, PyYAML, pytest).
- Run `pytest` from repository root to execute unit tests.
- Run `python -m src.train --cfg configs/default.yaml` to do a quick training run.

All hyperparameters are declared in `configs/default.yaml` and reflected by `src/config.py`.

6. Limitations and Extensions
-----------------------------
- The dataset is synthetic and meant for CI/training checks; replace it with env-based rollouts for RL experiments.
- The policy update uses a simple advantage (v - baseline) computed from the value network; more advanced methods (GAE, actor-critic loops, entropy regularization) could be added.

7. Code & Files
----------------
- `src/`: core modules (`config.py`, `data.py`, `model.py`, `losses.py`, `utils.py`, `train.py`)
- `configs/default.yaml`: default experiment configuration
- `tests/`: minimal tests ensuring importability and forward passes

8. References
-------------
- Sutton & Barto, "Reinforcement Learning: An Introduction" (for policy gradient concepts)

Appendix
--------
- Contact: maintainers of the course repository.
