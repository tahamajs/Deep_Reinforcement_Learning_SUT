# CA29 — Report: Soft Actor-Critic (SAC) Implementation

## Abstract

This report documents the SAC implementation in this repository, the experimental protocol to evaluate its performance on continuous control tasks, and templates for reporting and reproducing results. It contains theory, implementation details, exact hyperparameters, and step-by-step instructions to run experiments and produce publication-ready figures and tables.

---

## 1. Introduction & Goals

Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm that augments the RL return with an entropy bonus to encourage exploration. The goals of this project are:
- Provide a clean, import-safe, and well-tested PyTorch implementation of SAC.
- Supply a reproducible experiment pipeline and reporting templates for reproducible RL research.
- Demonstrate SAC's benefits over deterministic baselines through controlled experiments and ablations.

This report explains how to run experiments, what to report, and how to interpret results.

---

## 2. Background & Theory (concise)

### Maximum Entropy Objective
SAC optimizes the maximum entropy RL objective:

\[ J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right] \]

where \(\alpha\) controls the trade-off between reward maximization and entropy.

### Policy (Actor) Gradient (reparameterized)
Using the reparameterization trick with a squashed Gaussian policy,

\[ \nabla_\theta J_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \nabla_\theta \log \pi_\theta(a|s) \left( \alpha \log \pi_\theta(a|s) - Q_\phi(s,a) \right) \right] \]

in practice the algorithm uses
\(\alpha \log \pi_\theta(a|s) - Q_\phi(s,a)\) as the actor loss.

### Q-Function Update (Critic)
Learn Q by minimizing:

\[ J_Q(\phi) = \mathbb{E}_{(s,a,r,s')} \left[ \frac{1}{2} ( Q_\phi(s,a) - y )^2 \right], \]

where
\( y = r + \gamma ( \min_{i=1,2} Q_{\phi_{targ},i}(s', a') - \alpha \log \pi_\theta(a'|s') ) \).

### Temperature (\alpha) Tuning
We optionally tune \(\alpha\) automatically to match a target entropy \(\mathcal{H}_0\) (commonly \(-|\mathcal{A}|\)). The update minimizes
\( -\alpha ( \log \pi(a|s) + \mathcal{H}_0 ) \).

---

## 3. Implementation Notes (what to inspect)

- `src/config.py`
  - `SACConfig` dataclass centralizes experiment hyperparameters and supports YAML load/save.
  - Key fields: `env_name`, `gamma`, `alpha`, `automatic_entropy_tuning`, `target_entropy`, `lr_actor`, `lr_critic`, `batch_size`, `buffer_size`, `num_steps`, `eval_freq`, `tau`, `seed`, `device`, `log_dir`.

- `src/utils.py`
  - Seeding utilities (`set_seed`, `set_env_seed`) and deterministic-mode helpers (`make_deterministic`).
  - `get_device` returns CPU/CUDA intelligently.

- `src/sac.py`
  - `Actor`: squashed Gaussian policy with reparameterization and log-prob correction for `tanh` squashing.
  - `Critic`: twin Q-networks and target networks.
  - `ReplayBuffer`: efficient float32 storage; returns rewards/dones as column vectors for stable broadcasting.
  - `SAC.update()`: full update for critics, actor, and alpha; includes soft target updates (polyak averaging).
  - `save`/`load` methods save model/optimizer state and restore `log_alpha` robustly across devices.

- `src/experiment.py`
  - `Experiment` class handles env creation, seeding, training loop, evaluation, logging, and a simple `metrics.csv` writer.
  - Compatible with Gymnasium (handles reset/step signatures that return `(obs, info)` and `(obs, reward, terminated, truncated, info)`).

- `src/cli.py` provides a simple CLI to run experiments with config overrides.

- Tests live under `tests/` and cover config IO, utilities, replay buffer, networks, and a smoke test (disabled by default).

---

## 4. Experimental Protocol (detailed)

### Environments
Choose continuous control tasks to evaluate SAC:
- Recommended: `HalfCheetah-v4`, `Walker2d-v4`, `Hopper-v4`, `Ant-v4` (if MuJoCo/Gym-compatible engines available).

### Training Schedule
- Typical run: 1,000,000 environment steps
- Evaluation every `eval_freq` (default 10,000 steps) with `num_eval_episodes`=10
- Record metrics at each evaluation: average return, std, policy entropy, losses, and runtime

### Seeds & Statistical Protocol
- Use at least 5 random seeds per configuration (10 preferred for robust claims)
- Report mean ± std across seeds for final performance and include learning curves (mean ± std envelopes)
- For pairwise comparisons, report p-values (paired t-test or bootstrap CI)

### Ablations to Run
- Entropy ablation: `automatic_entropy_tuning` on vs off; fixed `alpha` values {0.01, 0.1, 0.5}
- Doubled critics: Compare single vs twin critics
- Network capacity: Hidden sizes {64, 256}
- Replay buffer size: {100k, 1M}
- Batch size sensitivity: {64, 256}

Each ablation should use the same seeds and evaluation protocol to be comparable.

---

## 5. Hyperparameters & Default Configs

A sample default hyperparameter set is provided in `configs/default.yaml`. Key values and suggested ranges:

| Name | Default | Suggested range | Notes |
|------|--------:|----------------:|-------|
| gamma | 0.99 | 0.95 - 0.999 | Discount factor |
| lr_actor | 3e-4 | 1e-4 - 1e-3 | Adam lr |
| lr_critic | 3e-4 | 1e-4 - 1e-3 | Adam lr |
| alpha | 0.2 | 1e-4 - 1.0 | Initial temperature if fixed |
| automatic_entropy_tuning | true | true/false | Enable adaptive alpha |
| batch_size | 256 | 64 - 512 | Larger batch sizes usually stabilize learning |
| buffer_size | 1e6 | 1e5 - 1e7 | Off-policy memory |
| tau | 0.005 | 0.001 - 0.01 | Polyak averaging factor |

Always list the full final configuration (YAML) when reporting results.

---

## 6. Evaluation Metrics & Visualization

Collect and visualize:
- Learning curves: `avg_reward` vs environment steps (plot mean ± std across seeds)
- Entropy trace: `mean_policy_entropy` vs steps
- Losses: actor/critic/alpha training losses vs steps (for stability analysis)
- Tables: final mean ± std at predetermined checkpoints (e.g., 100k, 200k, 1M steps)

Visualization best practices:
- Smooth noisy curves using low-pass smoothing (rolling window or exponential smoothing) but show raw curves in appendix
- Include confidence intervals (std or bootstrap CIs)
- Annotate important events (e.g., learning rate changes, algorithmic changes)

Example plotting code lives in `notebooks/experiment_template.ipynb` and reads `metrics.csv` in the experiment `log_dir`.

---

## 7. Reproducibility Checklist & Commands

Reproducibility checklist (complete for each reported experiment):
- [ ] Commit hash recorded (git rev-parse HEAD)
- [ ] Config YAML saved in results folder
- [ ] Seeds listed (and random number generator state saved if necessary)
- [ ] Environment versions recorded (Gym/Gymnasium, MuJoCo version if used)
- [ ] Checkpoint(s) and logs archived in `results/<run>/`

Common commands:
- Run experiment with defaults:
```bash
python -m src.cli --config configs/default.yaml --log-dir results/run1
```
- Evaluate a saved checkpoint (update `Experiment` helper or write a short script in `notebooks/` to load and evaluate)
- Run tests:
```bash
pytest -q
# Optional smoke test (slower):
RUN_SMOKE=1 pytest -q
```

---

## 8. Results Template & Example Table (fill in with your numbers)

### Example Final Results Table (filled example)

Below is a concrete, synthetic example of how you might fill in the final-results table after running full experiments (1M steps, 5 seeds). These numbers are placeholders to illustrate formatting and interpretation.

| Env | Method | Mean Return @1M (mean ± std) | Notes |
|-----|--------|------------------------------:|-------|
| HalfCheetah | SAC (adaptive α) | 4620 ± 185 | Rapid early learning; stable asymptotic performance |
| HalfCheetah | SAC (α=0.1 fixed) | 4350 ± 260 | Slightly slower convergence; more variance |
| HalfCheetah | SAC (α=0) | 3820 ± 510 | Lower final score, unstable early training |
| HalfCheetah | TD3 (baseline) | 4110 ± 300 | Competitive, but lower sample efficiency |
| Walker2d    | SAC (adaptive α) | 3220 ± 340 | Robust across seeds |
| Walker2d    | SAC (α=0.1 fixed) | 2980 ± 420 | Moderate sensitivity to α |
| Hopper      | SAC (adaptive α) | 2400 ± 150 | Fast convergence, low variance |

**Interpretation**: In this synthetic example, SAC with adaptive alpha achieves the best sample efficiency and final performance across environments. Fixed-alpha variants are more sensitive and show higher variance; turning off entropy ( = 0) damages both sample efficiency and final performance. TD3 performs reasonably but is outperformed by SAC in these cases.

### Demo Results (included example)

To help you verify plotting and report generation, this repository includes a small demo results folder with a synthetic `metrics.csv` that the notebook can read directly:

- `results/example_run/metrics.csv` — contains a few evaluation points with `step,avg_reward,std_reward` for a single demo run.
- Use `notebooks/experiment_template.ipynb` and set `run_dir='results/example_run'` to load and visualize these demo metrics.

```
# View the demo CSV (example)
cat results/example_run/metrics.csv
# Run the notebook and point to the demo run_dir
# or in Python:
import pandas as pd
pd.read_csv('results/example_run/metrics.csv')
```

### Figure Template
- Figure 1: Learning curves (avg return ± std) for baselines and variants
- Figure 2: Ablation plot (final score vs hyperparameter)
- Figure 3: Entropy trace and loss components

Include a short paragraph interpreting the results (what changed, why, whether differences are significant).

---

## 9. Limitations, Failure Modes, and Notes

- SAC may be sensitive to reward scaling and observation normalization: ensure environments use consistent reward scales.
- Deterministic seeding might not fully remove nondeterminism on GPU; report seeds and hardware.
- Some environments require MuJoCo or other licensed simulators—note this in the final report.

---

## 10. Code & Tests Summary

- Unit tests: `tests/test_config.py`, `tests/test_utils.py`, `tests/test_sac.py`, `tests/test_smoke.py` (skipped unless `RUN_SMOKE=1`).
- CI workflow: `.github/workflows/ci.yml` runs fast tests on push/pull requests.

---

## 11. Appendix — Detailed Reference

This appendix provides detailed references and auxiliary material that are useful when reproducing experiments or extending the codebase.

File-by-file detailed descriptions
- `configs/`:
  - `default.yaml`: Canonical defaults used for most experiments; contains values for learning rates, entropy settings, batch size, buffer size, `tau`, and logging directory.
  - `ant.yaml`: Example config for `Ant-v4` with a different seed and log directory.
- `src/config.py`:
  - Contains the `SACConfig` dataclass and `load_config` / `save_config` helpers. Use these to serialize/deserialize experiment settings.
- `src/utils.py`:
  - `set_seed`, `set_env_seed`, `get_device`, `make_deterministic` — ensure reproducible setup across runs.
- `src/sac.py`:
  - `Actor`, `Critic`, `ReplayBuffer`, and `SAC` training logic. The `SAC` class provides `select_action`, `update`, `save`, and `load` methods.
- `src/experiment.py`:
  - Provides the `Experiment` runner with training loop, evaluation, logging, and a simple `metrics.csv` writer. It is Gymnasium-compatible (supports varied reset/step signatures).
- `src/cli.py`:
  - Simple CLI for launching experiments with config overrides. Preferred entry point for reproducible runs.
- `notebooks/experiment_template.ipynb`:
  - Notebook for debugging, short experiments, and plotting; includes cells to load `metrics.csv` and generate publication-quality plots.
- `tests/`:
  - Unit tests for config I/O, utils, replay buffer, actor/critic shape tests, and an optional smoke test for training updates.

Logging, saved artifacts, and formats
- `training.log` — plain text logs written by Python `logging` (INFO level by default) and stored in experiment `log_dir`.
- `metrics.csv` — evaluation summary file appended during training (columns: `step,avg_reward,std_reward`). This file is used by notebooks to generate learning curves.
- `sac_final.pth` — PyTorch checkpoint saved by `SAC.save()` containing model states, optimizers, and `alpha`/`log_alpha` information.
- `results/<run_tag>/` — recommended directory layout for a single run:
  - `configs.yaml` or a copy of the used config file
  - `training.log`
  - `metrics.csv`
  - `sac_final.pth` and optional intermediate checkpoints
  - `plots/` — store PNG/PDF figures used in the report
  - `metadata.txt` — contains git commit hash, date, environment versions, and seed list (template below)

metadata.txt template (plain text)
```
commit: <git-sha>
date: 2025-12-19
config: configs/default.yaml
seeds: [42, 123, 456, 789, 101112]
environment: Gymnasium 0.26.0, MuJoCo (if used): 2.x
notes: brief notes about any modifications
```

Evaluation snippet: how to load a checkpoint and perform deterministic evaluation
```python
from src.config import SACConfig
from src.sac import SAC
from src.utils import get_device, set_env_seed
import gymnasium as gym

cfg = SACConfig()
device = get_device(cfg.device)
env = gym.make(cfg.env_name)
set_env_seed(env, cfg.seed + 1000)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SAC(state_dim, action_dim, cfg, device)
agent.load('results/run1/sac_final.pth')

# deterministic evaluation
rewards = []
for _ in range(10):
    obs, _ = env.reset()
    done = False
    ep_ret = 0
    while not done:
        a = agent.select_action(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_ret += r
    rewards.append(ep_ret)
print('avg', sum(rewards)/len(rewards))
```

Plotting & figure export guidelines (for publication-ready figures)
- Use vector formats (PDF/SVG) for line plots where possible and PNG for raster images.
- Recommended styling: use 2pt line width for main curves, 0.5 alpha fill for std bands, legible font sizes (12+), and colorblind-friendly palettes (e.g., `seaborn.color_palette('colorblind')`).
- Save figures at 300 DPI for raster export and verify fonts are embedded in PDFs for portability.

Hyperparameter sweep guidelines
- For small-scale sweeps, vary one hyperparameter at a time and seed each setting with multiple seeds.
- For large-scale sweeps, use consistent naming in `results/` (e.g., `results/halfcheetah_batch256_lr3e-4_seed42`) and save full configs alongside results.
- Use simple orchestration (bash loops, GNU parallel, or external schedulers); document resource usage (GPU #, memory) when reporting.

Submission checklist (use before final submission)
- [ ] All required code is present under `src/` and import-safe.
- [ ] `configs/` contains the configs used for main results.
- [ ] `results/` contains `metrics.csv`, saved checkpoints, and plots for main experiments.
- [ ] `REPORT.md` is updated with final numbers, figures paths, and commit hash.
- [ ] Unit tests pass locally (and on CI); consider running the optional smoke test before finalizing.
- [ ] Include a short README in `results/` describing how each figure/table was produced.

How to cite
- If you use this implementation for research, cite the original SAC paper:
  Haarnoja, T. et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018).

Licensing & acknowledgements
- This assignment codebase is provided under the repository license (see `LICENSE`). Do not include third-party proprietary code without proper attribution.

Contact & support
- For questions about this assignment, open an issue or contact the maintainers (see `AGENTS.md` for authorship notes).

---

## 12. How to Add Results to this REPORT

1. Create a `results/<tag>` directory and run the experiment with `--log-dir results/<tag>`.
2. After runs complete, copy aggregate metrics (CSV/plots) into `results/<tag>` and update the tables/figures in this REPORT with actual numbers and figure paths (e.g., `results/<tag>/learning_curve.png`).
3. Add the config YAML and `git rev-parse HEAD` output into `results/<tag>/metadata.txt`.

---

## 13. References

- Haarnoja, T. et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018).
- Other relevant works: DDPG, TD3, SAC variants.



---

_Last updated: commit / HEAD – add date and commit hash when publishing results._

