# CA24 Usage Guide

## Running experiments

Use `python -m src.experiment` for a quick demo run. For more controlled experiments, load YAML configs with `src.config.load_from_yaml` and call `src.experiment.run_experiment`.

## Extending the project

- Implement new `Dataset` classes under `src/data.py` or new module `src/datasets/`.
- Add new loss terms under `src/losses.py`.
- Keep notebooks as lightweight reproducible summaries of runs; heavy runs should be launched via scripts that record outputs to `outputs/`.
