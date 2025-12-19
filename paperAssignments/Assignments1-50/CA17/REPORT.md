# CA17 — Report

## Title
Simple Policy Gradient on CartPole-v1 (Baseline Implementation)

## Authors
Author: student@example.com

## Abstract
This report documents a minimal policy gradient baseline that implements a Monte-Carlo (REINFORCE) training loop with entropy regularization on the CartPole-v1 environment. The goal is to provide a clean, import-safe, and well-documented scaffold suitable for experimentation and teaching.

---

## 1. Introduction and Motivation
Balancing CartPole is a canonical control benchmark for reinforcement learning. CA17 provides a minimal, well-structured implementation so students can experiment with policy gradient ideas (variance reduction, entropy regularization, and training stability) without distractions from engineering complexity.

---

## 2. Related Work
- Williams (1992) — REINFORCE: policy gradient basics.
- Sutton & Barto — foundational textbook material on Monte-Carlo policy gradient methods.
- Modern improvements: GAE (Schulman et al.), PPO (Schulman et al.) for stable policy optimization.

---

## 3. Method
### 3.1 Environment
- CartPole-v1 (OpenAI Gym / Gymnasium). Observations are 4-dimensional continuous vectors; action space is discrete with 2 actions.

### 3.2 Policy Architecture
- `MLPPolicy` (in `src/model.py`): an MLP with two hidden layers and ReLU activations. The network outputs logits which are converted to a Categorical distribution for action sampling.
- Input dimension inferred from environment observation; output dimension equals `env.action_space.n`.

### 3.3 Loss and Optimization
- Policy gradient loss (REINFORCE): -E[log pi(a|s) * G], where G is the return for the sampled episode. Implemented in `src/losses.py` as `policy_gradient_loss`.
- Entropy regularization: `entropy_loss` returns -coeff * H(pi) to encourage exploration.
- Optimizer: Adam with learning rate defined in `src/config.py`.
- Returns computed per episode using discounted sum (Monte-Carlo). A simple mean subtraction is used as an advantage baseline.

### 3.4 Training Loop
Implemented in `src/train.py`:
- Collect one episode (or multiple depending on config).
- Compute discounted returns and advantages (returns - mean(returns)).
- Compute policy gradient loss and entropy loss, sum them, and perform a single update per episode.
- Periodically save checkpoints to `outputs/ca17` using `src/utils.py` helpers.

---

## 4. Experimental Design
### 4.1 Hyperparameters (default)
- env_name: CartPole-v1
- seed: 42
- lr: 1e-3
- hidden_size: 128
- gamma: 0.99
- total_timesteps: 50,000
- rollout_length: 2048 (unused in the simple loop but present for extensions)

A full hyperparameter sweep would vary learning rate, entropy coeff, and batch/rollout sizes.

### 4.2 Evaluation Metrics
- Episode Return (sum of rewards) averaged over evaluation episodes
- Learning curve: episode return vs steps
- Variance across seeds (report mean and standard deviation across 3–5 seeds)

### 4.3 Ablations (suggested)
- Remove entropy regularization to measure exploration effect
- Use advantage normalization or baseline networks
- Replace Monte-Carlo returns with GAE and compare sample efficiency

---

## 5. Reproducibility
- Deterministic seeds: `src/utils.set_seed` seeds Python, NumPy, and PyTorch (when available).
- Environment versions: the code aims to be compatible with Gymnasium >=0.26 and Gym; exact behavior of `env.reset()` and `env.step()` is handled in `src/data.py`.
- To reproduce experiments, follow the `How to reproduce` section below. Capture logs and checkpoints saved under `outputs/ca17`.

---

## 6. Results (Guidance for Reporting)
This scaffold does not include precomputed figures. When running experiments, report the following:
- Learning curves (mean ± std over seeds) for episode return.
- Final average return after 50k timesteps.
- A short table comparing ablations (entropy on/off, baseline/no-baseline).

Include sample command lines, e.g.,:

```bash
python -m paperAssignments.Assignments1_50.CA17.src.train --env CartPole-v1
```

For quick experiments, run the notebook `notebooks/CA17_experiment_template.ipynb` after reducing `total_timesteps` in the config.

---

## 7. Limitations and Future Work
- The current baseline uses single-episode updates and is high-variance.
- Scaling to continuous action spaces requires a Gaussian policy.
- Recommended future work: implement PPO, actor-critic with value baseline, GAE, or vectorized rollout collection.

---

## 8. Practical Notes and Artifacts
- Files of interest:
  - `src/config.py` — default hyperparameters
  - `src/model.py` — policy implementation
  - `src/losses.py` — loss helpers
  - `src/data.py` — episode collection helper
  - `src/utils.py` — seeding and checkpoint helpers
  - `src/train.py` — main training loop
  - `tests/` — unit tests and smoke tests
  - `notebooks/` — experiment template
- Checkpoints saved to `outputs/ca17` (create this directory or let the code create it at runtime).

---

## 9. How to reproduce
1. Create and activate a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install -r paperAssignments/Assignments1-50/CA17/requirements.txt
```

3. Run training (from repo root):

```bash
python -m paperAssignments.Assignments1_50.CA17.src.train
```

4. Run tests:

```bash
python -m pytest paperAssignments/Assignments1-50/CA17/tests
```

5. Re-run with different `--env` or modified `src/config.py` hyperparameters for ablations.

---

## 10. Appendix: Hyperparameter Table
| Name | Default | Notes |
|------|---------|-------|
| env_name | CartPole-v1 | Gym/Gymnasium environment |
| seed | 42 | randomness control |
| lr | 1e-3 | Adam learning rate |
| hidden_size | 128 | hidden units per layer |
| gamma | 0.99 | discount factor |
| total_timesteps | 50000 | total environment steps |


---

## 11. Contact
For questions about this implementation, contact the course staff or the author noted at the top of this report.

## Files
- `src/` — implementation
- `tests/` — unit tests
- `notebooks/` — suggested place for experiments (not included by default)

## License
This assignment falls under the repository license. Any code submissions should follow the project's contribution guidelines.
