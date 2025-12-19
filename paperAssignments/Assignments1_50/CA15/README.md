# CA15 — Curriculum Assignment 15

## Overview

CA15 is a compact, educational implementation demonstrating a small policy-gradient-style pipeline (MLP policy and value network). It is intentionally minimal and suitable for quick local experimentation, unit testing, and as a scaffold for extensions.

This folder contains:
- `src/` — import-safe Python modules (model, data, losses, utils, training script)
- `configs/` — example config (`default.yaml`)
- `notebooks/` — `quick_demo.ipynb` template (non-executed)
- `tests/` — minimal smoke tests for importability and forward passes
- `REPORT.md` — full report and reproduction instructions (read this first)

## Learning Goals

- Understand the basic policy-gradient training loop and value regression.
- Learn how to structure import-safe modules for research code.
- Gain hands-on experience with minimal reproducible experiments and testing.

## Quick Start

1. Create a virtual environment and install dependencies (Python 3.10+):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pyyaml matplotlib pytest
```

2. Run tests (from repository root):

```bash
python -m pytest paperAssignments/Assignments1_50/CA15/tests -q
```

3. Run a short CPU training example:

```bash
python -m src.train --cfg configs/default.yaml
```

Or programmatically:

```python
from config import Config
from train import train
res = train(Config(epochs=2))
print(res)
```

## Files & Usage (summary)

- `src/config.py` — `Config` dataclass and `Config.from_yaml(path)`
- `src/data.py` — `SyntheticDataset(input_dim, output_dim, size, seed)` + `batches(batch_size)`
- `src/model.py` — `MLPPolicy`, `ValueNetwork` ; `MLPPolicy.get_action(x, deterministic=False)`
- `src/losses.py` — `mse_loss`, `policy_gradient_loss`
- `src/utils.py` — `set_seed`, `save_checkpoint`, `load_checkpoint`
- `src/train.py` — `train(cfg, save_path=None)` and CLI

See `REPORT.md` for a full file-by-file API description and examples.

## Notebook Template

Open `notebooks/quick_demo.ipynb` to see a guided, non-executed template that walks through imports, config loading, data sampling, model instantiation, a single training step, and checkpointing examples.

## Reproducibility

- Use `utils.set_seed(seed)` at the start of experiments.
- Save `cfg.__dict__` in `save_checkpoint(..., extra={"cfg": cfg.__dict__})` for provenance.
- For exact reproducibility, pin dependency versions (use a `requirements.txt`) and record `torch.__version__` and platform information.

## Troubleshooting

- Import errors in tests: ensure `PYTHONPATH` includes the CA `src/` directory (tests set this up normally). If running ad-hoc, `export PYTHONPATH=paperAssignments/Assignments1_50/CA15/src`.
- CUDA errors: switch to CPU by editing `configs/default.yaml` and setting `device: cpu`.
- NaN losses: reduce learning rate, check input data ranges, or overfit a tiny batch to sanity-check gradients.

## Contributing

- Make small, focused changes scoped to this CA.
- Add tests for behavior you change or add.
- Update `REPORT.md` and this `README.md` when APIs change.

## License & Citation

- See the repository `LICENSE` at the root for license terms. If you reuse parts of this CA, include a citation to the course or project repository.

## Contact

Open an issue or PR in the main course repository; add a short description and a minimal reproduction if you expect code changes.















