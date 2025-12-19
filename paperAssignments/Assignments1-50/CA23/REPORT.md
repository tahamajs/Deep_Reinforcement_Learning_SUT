# CA23 — Report

## Abstract

This short report documents the CA23 scaffold: design decisions, proposed
experiments, and reproducibility instructions. The repository provides simple
policy and value networks, loss utilities for policy-gradient algorithms, and
minimal data utilities for on-policy rollouts. The goal is to enable quick
experiment iteration and reproducible reporting for course projects.

## Goals and Hypothesis

- Goal: Evaluate a baseline policy-gradient agent against a simple environment
  benchmark and investigate the effect of entropy regularization and
  network capacity on learning stability.
- Hypothesis: Moderate entropy regularization improves exploration in early
  training and leads to more stable returns across seeds; larger networks
  reduce bias at the cost of increased variance.

## Methods / Implementation

- Networks: `PolicyNetwork` (categorical) and `ValueNetwork` (scalar) in
  `src/model.py` with configurable hidden sizes.
- Losses: standard policy gradient (`policy_gradient_loss`), MSE value loss
  (`value_loss`), and entropy term (`entropy_loss`) in `src/losses.py`.
- Data: `collect_episodes` for rolling out policies and `discounts` for return
  computation in `src/data.py`.
- Utilities: seeding (`set_seed`), device selection (`get_device`), and
  figure saving (`save_figure`).

## Suggested Experiments

1. Baseline training on `CartPole-v1` using `ExperimentConfig` default values.
2. Entropy sweep: train with entropy coefficients {0.0, 0.01, 0.1} and compare
   mean returns across 5 seeds.
3. Network size ablation: hidden sizes `[(32,32), (64,64), (128,128)]`.
4. Learning rate sensitivity: sweep {1e-4, 1e-3, 3e-3}.

For each run, record per-episode return, loss terms, and optionally save
checkpoints and figures.

## Evaluation Metrics & Reporting

- Primary metric: average episode return (smoothed by moving average, e.g.,
  window=50).
- Secondary metrics: policy/value losses, entropy, and sample efficiency
  (episodes to threshold).

Suggested figures:
- Learning curves (return vs. episodes) with shaded error over seeds.
- Bar charts comparing final performance across hyperparameters.

Suggested tables:
- Mean ± std return at convergence for each hyperparameter setting.

## Reproducibility Checklist

- Record the exact YAML config used (store under `runs/<id>/config.yaml`).
- Fix RNG seeds (call `set_seed` at start of training) and record seeds.
- Log random seeds and software/hardware environment (Python, torch, CPU/GPU).

## How to extend

- Add Advantage estimators (GAE), baselines, or actor-critic loops into
  `scripts/train.py` or dedicated notebooks.
- Add more robust evaluation (e.g., multiple environments and deterministic
  evaluation runs).

## Notes & Caveats

- This scaffold purposely keeps dependencies minimal and keeps heavy runtime
  out of imports so unit tests remain fast.
- For research-grade experiments, add checkpointing, structured logging
  (e.g., CSV/JSONL), and a more detailed experiment runner.

## Experimental Protocol (detailed)

This section describes a reproducible experiment protocol for running the
recommended sweeps. Use this as the canonical reference when reproducing the
paper-style results.

1. Configuration and versioning
   - Create a `runs/<experiment-id>/` folder for each experiment.
   - Save the **complete** YAML config used as `config.yaml` (use
     `ExperimentConfig.to_yaml`). Record the git commit hash and the
     environment (Python, torch versions).

2. Seeds and replicates
   - Run each hyperparameter setting across N=5 different seeds (e.g., 0,1,2,3,4).
   - Use `set_seed(seed)` at the start of each run and record the seed in the
     log file.

3. Logging
   - Log per-episode returns, policy loss, value loss, entropy, and learning
     rate to a CSV file (`runs/<id>/metrics.csv`).
   - Save model checkpoints periodically under `runs/<id>/checkpoints/`.

4. Compute resources
   - Small experiments (CartPole) can run on CPU; for larger experiments use
     GPU and record device IDs in the run metadata.

5. Postprocessing and plots
   - Produce learning curves with shaded confidence intervals across seeds
     (e.g., mean ± std or SEM). Use `save_figure` to store high-resolution
     figures (`dpi=300`).

---

## Metrics, Statistical Tests & Result Reporting

- Report mean ± standard deviation across seeds for final episode return at
  convergence (or after a fixed number of episodes).
- If comparing two conditions, perform a two-sided t-test on the final
  returns across seeds, reporting p-values and effect sizes (Cohen's d).
- When presenting plots, include shaded error bands and mark significant
  differences with standard notation (*p* < 0.05, **p* < 0.01).

---

## Example hyperparameter table (template)

| Experiment | lr | hidden | entropy | seeds | mean return | std |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1e-3 | (64,64) | 0.0 | 5 |  |  |
| entropy0.01 | 1e-3 | (64,64) | 0.01 | 5 |  |  |
| large-net | 1e-3 | (128,128) | 0.0 | 5 |  |  |

---

## Algorithm Pseudocode (vanilla policy gradient)

1. Initialize policy π_θ and value V_φ with random weights.
2. For each episode:
   a. Collect trajectory τ = (s_0, a_0, r_0, ..., s_T).
   b. Compute returns R_t = sum_{k≥t} γ^{k-t} r_k.
   c. Compute advantages A_t = R_t − V_φ(s_t).
   d. Update policy θ ← θ + α * ∇_θ E_t[log π_θ(a_t|s_t) * A_t] (SGD step with
      gradient ascent implemented as minimizing −E[log π * A]).
   e. Update value parameters φ by minimizing mean-squared error between
      V_φ(s_t) and R_t.
   f. Optionally add entropy regularization term to the policy objective.

---

## File map and responsibilities

- `src/config.py` — `ExperimentConfig`: centralizes hyperparameters and YAML
  serialization (validation in `__post_init__`).
- `src/model.py` — `PolicyNetwork`, `ValueNetwork`: MLP-based modules with
  configurable hidden sizes. `PolicyNetwork.get_action` returns actions and
  log-probabilities.
- `src/losses.py` — `policy_gradient_loss`, `value_loss`, `entropy_loss`: small
  utilities used by scripts and tests.
- `src/data.py` — `collect_episodes` (lazy `gym` import), `discounts`:
  lightweight rollouts and return computation.
- `src/utils.py` — seeding, device helpers, `ensure_dir`, `save_figure`.
- `scripts/train.py` — Example training loop (pedagogical; not optimized for
  large experiments). Use it as a starting point for more sophisticated
  runners.
- `configs/` — Example YAMLs for quick experiments (`default.yaml`,
  `debug.yaml`).
- `tests/` — Unit tests for core components to ensure import-safety and
  numerical sanity checks.

---

## Reproducibility checklist (detailed)

- [ ] Save the full YAML config for each run.
- [ ] Save the git commit hash and diff of relevant files.
- [ ] Record package versions (e.g., `pip freeze > requirements.txt`).
- [ ] Save RNG seeds and ensure `set_seed` is used at run startup.
- [ ] Save per-episode logs and periodic checkpoints.
- [ ] Document compute hardware and OS.

---

## Limitations & Ethical Considerations

- Small benchmarks (CartPole) may not generalize to complex tasks; do not
  over-claim empirical findings without larger-scale validation.
- Ensure any downstream research follows data usage and compute guidelines.

---

## Acknowledgements & Reuse

This scaffold is provided for educational use. If you use parts of this
scaffold in published work, please cite the course and maintainers.

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction" (for PG/advantage).
- OpenAI Spinning Up: implementation references and best practices.
