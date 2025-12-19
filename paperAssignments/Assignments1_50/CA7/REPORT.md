CA7 — Report
==============

Authors: Curriculum Assignment 7 (template)

Abstract
--------
This short report documents the minimal algorithmic components, experimental setup, and reproducibility instructions for CA7. The included implementation provides a Soft Actor-Critic (SAC) agent with MLP actor, critic, and temperature networks, a replay buffer, and utilities for training on continuous control tasks. This assignment focuses on off-policy RL with entropy regularization, reproducibility, clarity, and clean, import-safe API design.

1. Introduction
---------------
The goal of CA7 is to provide a compact, well-documented SAC implementation suitable for educational and research-first use. SAC combines soft Q-learning with a stochastic actor and automatic entropy tuning for sample-efficient, stable learning in continuous action spaces. The method includes twin critics, target networks, and replay buffers for stability.

2. Methods
----------
- Model: MLP actor (Gaussian policy), twin MLP critics (Q-functions), and a temperature network for entropy tuning.
- Losses: MSE for critics, policy gradient with entropy regularization, and log-prob for temperature.
- Data: Replay buffer with uniform sampling; synthetic or environment-based transitions.

3. Experimental Plan
--------------------
Although this repository does not include large-scale experiments, the intended small experiments are:

- Vary seed and compare final returns to ensure determinism.
- Vary alpha (temperature) and learning rates to observe exploration vs exploitation trade-offs.
- Use the training script `train_sac_full.py` to perform short CPU runs (e.g., steps=1e4) and collect `actor_loss`, `critic_loss`, `alpha`, and returns.

4. Expected Results and Evaluation
---------------------------------
- The training loop should complete without errors on CPU/GPU.
- On synthetic tasks, expect Q-values to converge and policy to improve modestly; on real envs (e.g., MuJoCo), expect increasing returns over time.
- Evaluate stability (losses decrease, no NaNs) and deterministic behavior across seeding.

5. Reproducibility
------------------
To reproduce results:

- Create a venv and install dev dependencies (PyTorch, PyYAML, pytest, gym).
- Run `pytest` from repository root to execute unit tests.
- Run `python train_sac_full.py --device cpu --out outputs/ca7_smoke` for a quick run.

All hyperparameters are declared in `src/config.py`.

6. Limitations and Extensions
-----------------------------
- The implementation is for continuous actions; extend to discrete with SAC-Discrete.
- Add prioritized replay or image inputs for advanced use cases.
- The replay buffer is basic; add HER for goal-conditioned tasks.

7. Code & Files
----------------
- `src/`: core modules (`config.py`, `data.py`, `model.py`, `losses.py`, `sac.py`, `utils.py`)
- `train_sac_full.py`: training script
- `notebooks/demo.ipynb`: demo notebook
- `tests/`: unit tests ensuring importability and forward passes

8. References
-------------
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (ICML 2018)

Appendix
--------
- Contact: maintainers of the course repository.