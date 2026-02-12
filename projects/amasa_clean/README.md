# AMASA-Complex v2 (Advanced RL + Safety)

AMASA-Complex v2 is a CPU-first, advanced course-scale reinforcement learning project for autonomous suturing. It keeps legacy script entrypoints while adding a config-driven experiment system, multiple algorithms, hybrid safety controls, benchmark orchestration, and IEEE reporting.

## What is new in v2

- **4-method algorithm suite**:
  - Offline: CQL, IQL
  - Online safe: SAC-Lagrangian, PPO-Lagrangian
- **Hybrid safety stack**:
  - PID Lagrangian dual update
  - Decision-tree action shield
  - Runtime trajectory risk critic gate
- **3 scenario families**:
  - `nominal`, `perturbed`, `adversarial`
- **YAML experiment system**:
  - base config + algorithm/scenario/preset overlays
- **Benchmark + sweep utilities**:
  - algorithm x scenario x seed matrix
  - PID grid (`kp`, `kd`) analysis + aggregated Pareto/heatmap
- **Backward-compatible CLI**:
  - existing scripts still work and now accept `--config`, `--preset`, `--scenario`, `--algo`

## Project layout

- `amasa/core/config.py` - YAML loading, merge, validation
- `amasa/core/registry.py` - environment/algorithm/safety registries
- `amasa/core/metrics.py` - summary and record exporters
- `amasa/envs/suturing_base.py` - configurable suturing dynamics
- `amasa/envs/scenarios.py` - nominal/perturbed/adversarial wrappers
- `amasa/envs/curriculum.py` - phase scheduling helper
- `amasa/offline/cql.py` - conservative Q-learning
- `amasa/offline/iql.py` - implicit Q-learning
- `amasa/online/sac_lagrangian.py` - constrained SAC baseline
- `amasa/online/ppo_lagrangian.py` - constrained PPO baseline
- `amasa/safety/risk_critic.py` - runtime risk model
- `amasa/safety/guard.py` - composed safety guard
- `amasa/bench/runner.py` - benchmark and PID job builders
- `amasa/bench/aggregate.py` - cross-run plotting/aggregation
- `scripts/run_experiment.py` - unified mode runner
- `configs/` - base + overlays + thresholds
- `tests/` - unit + smoke tests

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r /Users/tahamajs/Documents/uni/DRL/projects/amasa_clean/requirements.txt
```

## Legacy-compatible commands

```bash
python3 -m projects.amasa_clean.scripts.generate_dataset --episodes 50 --out data/amasa_offline.npz
python3 -m projects.amasa_clean.scripts.train_offline --dataset data/amasa_offline.npz --algo cql --preset smoke
python3 -m projects.amasa_clean.scripts.train_safe_online --algo sac_lag --preset smoke --use_shield
python3 -m projects.amasa_clean.scripts.train_hierarchical --episodes 20 --offline_checkpoint checkpoints/cql_final.pt
python3 -m projects.amasa_clean.scripts.evaluate --checkpoints checkpoints --scenario nominal --out plots/pareto.png
```

## Unified runner

```bash
# Offline run (config-driven)
python3 -m projects.amasa_clean.scripts.run_experiment \
  --mode offline_train --config projects/amasa_clean/configs/base.yaml --algo iql --scenario perturbed

# Online safe run
python3 -m projects.amasa_clean.scripts.run_experiment \
  --mode online_train --preset smoke --algo sac_lag --scenario adversarial \
  --checkpoint projects/amasa_clean/results/some_offline_run/checkpoints/iql_final.pt

# Benchmark matrix (CQL/IQL offline + SAC/PPO online)
python3 -m projects.amasa_clean.scripts.run_experiment \
  --mode benchmark --config projects/amasa_clean/configs/base.yaml --out_dir projects/amasa_clean/results/bench

# PID sweep
python3 -m projects.amasa_clean.scripts.run_experiment \
  --mode pid_sweep --config projects/amasa_clean/configs/base.yaml --algo sac_lag --out_dir projects/amasa_clean/results/pid

# Report build bundle
python3 -m projects.amasa_clean.scripts.run_experiment --mode report_bundle
```

## Acceptance thresholds

Tiered thresholds live in `configs/thresholds.yaml`:

- Nominal: min `(reward>100, cost<0.60)`, target `(reward>300, cost<0.35)`, stretch `(reward>600, cost<0.20)`
- Perturbed: min `(reward>50, cost<0.70)`, target `(reward>200, cost<0.45)`, stretch `(reward>450, cost<0.30)`
- Adversarial: min `(reward>0, cost<0.80)`, target `(reward>120, cost<0.55)`, stretch `(reward>250, cost<0.40)`
- Offline gate: finite losses + reward ratio vs random baseline >= 2.0

## Output artifacts

Default outputs are organized under run directories:

- `checkpoints/` - algorithm and shield checkpoints
- `summary.csv` - run-level metrics
- `results/*_summary.csv` - benchmark/sweep combined tables
- `plots/global_pareto.png` - aggregated frontier
- `plots/pid_heatmap.png` - sweep heatmap

## Notes

- CPU-first defaults are provided via `presets/smoke.yaml`.
- Use `presets/full.yaml` for extended runs and report figures.
- Existing script CLI remains valid while adding optional config-based controls.
