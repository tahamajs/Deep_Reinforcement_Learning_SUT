# REPORT — CA19: Actor–Critic with Value-Ensemble Uncertainty Bonus

**Authors:** Curriculum Assignment template
**Date:** (fill in)

---

## 1. Abstract
We implement an Actor–Critic baseline extended with a small ensemble of scalar value heads to estimate epistemic uncertainty. We use the variance across ensemble value predictions as an additive *uncertainty bonus* to the advantage (weighted by beta). This document describes the method, implementation details, experimental protocol, expected plots and a template for writing up results suitable for a short paper.

---

## 2. Introduction & Motivation
Exploration is a core challenge in reinforcement learning. Ensemble-based uncertainty estimates have been used effectively in model-free and model-based RL to prioritize learning and exploration. This work evaluates a simple, low-overhead approach: add a variance-based bonus to the policy gradient advantage to encourage exploration of epistemically uncertain states.

---

## 3. Method (detailed)
### 3.1 Notation
- s: state, a: action, r: reward, \gamma: discount factor.
- V_m(s): value predicted by ensemble member m (m=1..M).
- V̄(s) = (1/M) \sum_m V_m(s) (mean across ensemble).
- Var[V(s)] = (1/M) \sum_m (V_m(s) - V̄(s))^2.

### 3.2 Model architecture
- Shared trunk: MLP(obs_dim -> hidden_dim -> hidden_dim).
- Policy head: linear output of size action_dim (logits).
- Value heads: M scalar linear heads (
  hidden_dim -> 1), assembled as an `nn.ModuleList`.

Shapes:
- obs: (B, obs_dim)
- logits: (B, A)
- values: (M, B)

### 3.3 Loss functions and updates
- Critic target (one-step TD):
  target(s) = r + \gamma * V̄(s') * (1 - done)
- Critic loss: L_c = MSE(V̄(s), target(s))
- Actor augmented advantage: A'(s,a) = (target - V̄(s)) + beta * Var[V(s)]
- Actor loss: L_a = -E[A'(s,a) * log \pi(a|s)]

Training loop (per minibatch):
1. Compute logits and ensemble values V_m(s).
2. Compute next-state ensemble mean and target.
3. Update critic via MSE on mean values.
4. Recompute values for current batch, compute Var[V(s)].
5. Compute advantages and L_a with beta-weighted var term.
6. Update actor with gradient of L_a.

(For numerical stability, clamp or normalize var if needed; detach terms where required.)

---

## 4. Implementation details
### 4.1 Files and responsibilities
- `src/config.py`: `CAConfig` dataclass with default hyperparameters.
- `src/model.py`: `ActorCriticEnsemble` with `forward()` and `act()` APIs.
- `src/losses.py`: `critic_loss`, `actor_loss`, `value_ensemble_variance`.
- `src/data.py`: `ReplayBuffer` and Transition type for simple sampling.
- `src/utils.py`: seeding, checkpoint save/load helpers, `to_device`.
- `src/train.py`: debug training loop, plotting, and checkpointing.
- `src/experiment.py`: sweep helper that runs multiple seeds and saves CSV metrics.
- `scripts/aggregate_results.py`: basic aggregator to compute final return mean/std.
- `tests/`: pytest suites to validate shapes and determinism.

### 4.2 Hyperparameters (defaults)
| Name | Default | Notes |
|---|---:|---|
| seed | 0 | random seed |
| lr | 3e-4 | Adam learning rate |
| batch_size | 128 | minibatch for updates |
| gamma | 0.99 | discount |
| ensemble_size | 3 | number of value heads |
| hidden_dim | 64 | trunk hidden size |
| beta | 0.1 | weight for uncertainty bonus |
| total_steps | 2000 | steps in debug run |

For rigorous experiments increase `total_steps` substantially (e.g., 50k-500k depending on env).

### 4.3 Determinism & checkpointing
- Use `src/utils.set_seed(seed)` at the start to set Python, NumPy, Torch seeds and cuDNN deterministic flags (where available).
- `save_checkpoint` writes atomically (temporary file then move) to avoid partial files.
- Always save `configs/*.yaml` and `git` hash with run outputs.

---

## 5. Experimental protocol (reproducible)
### 5.1 Environment selection
- For initial debugging: `CartPole-v1` (fast, deterministic-ish, discrete actions).
- For more robust claims: add at least one continuous control environment (MuJoCo, Brax, or classic control tuned) and optionally more challenging sparse tasks.

### 5.2 Sweep and seeds
- For each (beta, ensemble_size), run N=5 independent seeds.
- Seeds should be saved and experiments run in a deterministic order to help reproducibility.

### 5.3 Logging & metrics
- Save per-episode metrics as CSV with columns:
  - timestamp, step, seed, episode, train_return, eval_return (optional), loss_actor, loss_critic, lr
- Save final checkpoint and a `meta.json` with `git` commit, config used, and command-line.

### 5.4 Aggregation
- After all runs finish: use `scripts/aggregate_results.py` or a Jupyter notebook to compute mean and std curves.
- For plotting: align episodes (or fixed-step bins). Compute mean and +/- std to make shaded curves.

### 5.5 Statistical testing
- For final comparisons (e.g., baseline beta=0 vs beta>0), run paired comparisons across seeds: bootstrap confidence intervals or paired t-test on final returns. Report effect size and p-values.

---

## 6. Plotting and reporting templates
### 6.1 Example plotting snippet (matplotlib)
```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# read CSVs in outputs/my_run
files = sorted(Path('outputs/my_run').glob('metrics_seed_*.csv'))
all_returns = []
for f in files:
    df = pd.read_csv(f)
    all_returns.append(df['train_return'].values)

# pad to same length
L = max(len(r) for r in all_returns)
arr = np.array([np.pad(r, (0, L-len(r)), 'edge') for r in all_returns])
mean = arr.mean(0)
std = arr.std(0)
plt.plot(mean)
plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Learning curve (mean ± std)')
plt.savefig('pictures/fig_01_reward.png', dpi=300)
```

### 6.2 Figures to produce
- Figure 1: Learning curves (mean ± std) for beta values (baseline vs variants).
- Figure 2: Bar chart of final return (mean ± std) across (beta, ensemble_size).
- Optional: State-wise variance heatmap / histogram of ensemble variance to show where uncertainty concentrates.

### 6.3 Table template (example)
| config | ensemble_size | beta | final_return_mean | final_return_std |
|---|---:|---:|---:|---:|
| baseline | 3 | 0.0 | 195.3 | 3.2 |
| beta=0.01 | 3 | 0.01 | 197.2 | 2.9 |

---

## 7. Results section template (fill after running experiments)
### 7.1 Quantitative results
- Describe dataset of runs (env, seeds, steps).
- Report mean and std final returns; include table above.
- State whether differences are statistically significant (report p-values and CI).

### 7.2 Qualitative analysis
- Plot example trajectories or episode summaries. Discuss how uncertainty bonus changes exploration patterns (e.g., visits to infrequently-visited states).
- Report ablation results (vary ensemble_size, beta) and interpret trends.

### 7.3 Limitations
- Discuss generalization beyond tested environments, sensitivity to beta, and compute overhead from ensemble heads.

### 7.4 Conclusion
- Short paragraph: summarize whether the uncertainty bonus improved sample efficiency, under what settings, and suggestions for future improvements.

---

## 8. Reproducibility checklist (detailed)
- [ ] Commit and push all code and configs used for experiments.
- [ ] Save `outputs/<run_id>` including `metrics_seed_*.csv`, `checkpoint_*.pt`, `meta.json`.
- [ ] Record environment: OS, Python version, PyTorch version, GPU (type), random seeds.
- [ ] Provide scripts for aggregation and plotting.

---

## 9. Appendix — Hyperparameter table & recommended ranges
| param | default | recommended grid |
|---|---:|---|
| lr | 3e-4 | {1e-4, 3e-4, 1e-3} |
| batch_size | 128 | {32, 64, 128} |
| beta | 0.1 | {0.0, 0.01, 0.1} |
| ensemble_size | 3 | {1, 3, 5} |
| total_steps | 2000 (debug) | {50k, 100k, 500k} |

---

## 10. Notes for authors / instructors
- This repository is designed as a teaching baseline. Students should extend `src/` and add clear tests for any new code.
- For a short assignment report, include: 1 paragraph method, 1 paragraph experiment details, 2 small figures, and 1 paragraph discussion.

---

**End of report template.**

_Fill in numerical results, attach figures from `pictures/`, and replace placeholders before submitting or publishing._