# Research-Grade Deep RL Framework (Clean Project Root)

This project is a from-scratch, multi-chain DRL framework designed for graduate research workflows under practical laptop constraints.

## What is implemented

- 5 chains, one unified interface:
  - `value`: `dqn`, `rainbow_lite`
  - `policy`: `reinforce`, `ppo`, `trpo_lite`, `cpo_lite`
  - `actor_critic`: `a2c`, `sac`
  - `model_based`: `dyna_q`, `mbpo_lite`
  - `marl`: `ippo`, `qmix_lite`
- Standard result schema in JSON for all runs
- Suite runner with seed sweeps and smoke/full budgets
- Plot and aggregate table generation
- Auto-filled English + Persian LaTeX reports

## Project layout

- `grad_rl/core/`: config, buffers, networks, metrics, logging, schedules
- `grad_rl/algorithms/*`: algorithm implementations by chain
- `grad_rl/envs/`: environment adapters/utilities
- `scripts/run_experiment.py`: unified single-run entrypoint
- `scripts/run_suite.py`: benchmark matrix runner
- `scripts/generate_plots.py`: figures + aggregate CSV
- `scripts/build_report_tables.py`: LaTeX table fragments from aggregate CSV
- `configs/chains/*.yaml`: chain+algo default configs
- `configs/suites/fast_5chain.yaml`: full benchmark matrix (3 seeds)
- `tests/`: unit and smoke tests

## Installation

```bash
cd /Users/tahamajs/Documents/uni/DRL/clean_grad_rl
python3.10 -m venv .venv_runs
source .venv_runs/bin/activate
pip install -r requirements.txt
```

## Single experiment

```bash
PYTHONPATH=. python scripts/run_experiment.py \
  --chain value \
  --algo dqn \
  --env CartPole-v1 \
  --steps 20000 \
  --seed 0
```

Output location pattern:

`outputs/runs/<chain>/<algo>/<env>/seed_<n>/metrics.json`

## Suite execution

Smoke mode:

```bash
PYTHONPATH=. python scripts/run_suite.py \
  --suite configs/suites/fast_5chain.yaml \
  --mode smoke
```

Full mode:

```bash
PYTHONPATH=. python scripts/run_suite.py \
  --suite configs/suites/fast_5chain.yaml \
  --mode full
```

## Analytics and report sync

```bash
PYTHONPATH=. python scripts/generate_plots.py \
  --suite-dir outputs/suite_reports \
  --runs-root outputs/runs

python scripts/build_report_tables.py \
  --aggregate-csv outputs/suite_reports/tables/aggregate_metrics.csv \
  --out-root .
```

Generated artifacts:

- Figures: `outputs/suite_reports/figures/*.png`
- Aggregate CSV: `outputs/suite_reports/tables/aggregate_metrics.csv`
- Table fragments: `report_table_*.tex`

## Compile reports

English:

```bash
latexmk -pdf university_project.tex
```

Persian:

```bash
latexmk -pdf report_fa.tex
```

## Tests

Fast unit tests:

```bash
PYTHONPATH=. pytest tests -k "not slow"
```

Smoke integration test:

```bash
PYTHONPATH=. pytest tests/test_integration_smoke.py -m slow
```

## Notes

- This root (`clean_grad_rl`) is the canonical implementation target.
- Tracking is local-only (JSON/CSV/PNG).
- Headline results should use 3 seeds (`0,1,2`) via suite config.
