# CA19 — Curriculum Assignment 19 ✅

## Overview

This package implements an Actor–Critic baseline extended with a value-ensemble uncertainty bonus for exploration. The goal of CA19 is to provide a small, well-tested, and reproducible research codebase that students can extend and run sweep experiments with reproducible seeds and logging.

---

## Quick start 🔧

1. Create and activate a virtualenv (Python 3.10+ recommended):

   ```bash
   python -m venv .venv && source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt  # if provided, or install torch, gymnasium, numpy
   ```

2. Run tests:

   ```bash
   cd paperAssignments/Assignments1-50/CA19
   python -m pytest -q
   ```

3. Run a short demo (CPU):

   - Open `notebooks/demo.ipynb` and run the cells (it runs a small debug training with `total_steps=200`).
   - Or run: `python -c "from src.config import CAConfig; from src.train import run_training; cfg=CAConfig(); cfg.total_steps=200; cfg.device='cpu'; run_training(cfg)"`

---

## Project layout 📁

- `src/` — implementation modules:
  - `config.py` — configuration dataclass `CAConfig`.
  - `model.py` — `ActorCriticEnsemble` model: policy + value ensemble.
  - `data.py` — small `ReplayBuffer` and `Transition` container.
  - `losses.py` — `critic_loss`, `actor_loss`, `value_ensemble_variance`.
  - `utils.py` — seeding, `to_device`, checkpoint helpers.
  - `train.py` — small training loop with plotting and checkpointing.
- `tests/` — pytest tests covering imports, forward shapes, losses, and utils.
- `notebooks/demo.ipynb` — short demo showing how to run a debug experiment.
- `configs/` — `debug.yaml` and `default.yaml` for reference configs.
- `pictures/` and `outputs/` — runtime output folders (created by training).

---

## Implementation summary (what's inside) 💡

- Actor-Critic architecture with a shared trunk and an ensemble of scalar value heads.
- Uncertainty bonus: variance across ensemble values per state, used as an additive term to the advantage with weight `beta`.
- Training script performs on-policy-like updates from a replay buffer (small debug implementation) and saves periodic checkpoints.

---

## Reproducing experiments and reporting 📊

### Experiment plan

- Baseline: beta=0.0 (no uncertainty bonus).
- Variant A: beta in {0.0, 0.01, 0.1}.
- Variant B: ensemble_size in {1, 3, 5}.

For each configuration: run 5 independent seeds, save per-episode returns to CSV (see `src/experiment.py` helper), and compute mean/std curves.

### Expected outputs

- `outputs/<run_id>/metrics.csv` with columns: timestamp, step, seed, episode, train_return, eval_return (optional), loss_actor, loss_critic, lr
- `pictures/fig_01_reward.png` — reward curve for the run
- `report/` — aggregated figures and a short writeup in `REPORT.md`

---

## Tests ✅

Run `python -m pytest -q`. Current tests:

- `tests/test_imports.py` — ensures modules import cleanly.
- `tests/test_forward.py` — checks model output shapes and `act()` signatures.
- `tests/test_loss.py` — ensures losses return finite scalars.
- `tests/test_utils.py` — (added) tests seeding determinism and checkpoint save/load.

---

## How to extend for a paper or assignment ✍️

1. Implement your variant in `src/` and add unit tests under `tests/`.
2. Add a configuration to `configs/` or a new YAML file for sweep automation.
3. Add a short notebook or script that runs the full sweep (5 seeds) and saves results to `outputs/`.
4. Aggregate results and create the figures for `REPORT.md`.
5. Write a one-page report (see `REPORT.md` template added in the repo) describing method, experiment, and interpretation.

---

## Practical notes & troubleshooting ⚠️

- If training diverges: try reducing `lr`, decreasing `beta`, or clipping gradients.
- When using GPU: set `cfg.device='cuda'` and use small batch sizes for low VRAM.
- For reproducibility: run with deterministic flags in `src/utils.set_seed` (it sets cudnn options where applicable).

---

## Citation / Licensing

This educational code is released under the repository license (see top-level `LICENSE`). Please cite the course or assignment if you use it as a baseline in published work.

---

## Contact and contributions

If you make improvements, please add tests and update `REPORT.md` with new results and expected figures.

---

(End of README)















