CA15 — Report
==============

**Authors:** Curriculum Assignment 15 (template)

Abstract
--------
This report documents the CA15 reference implementation, experimental setup, API, and reproducibility instructions. The implementation is intentionally minimal and import-safe: it contains a small MLP policy and value network, a synthetic dataset for quick checks, loss functions, utilities for checkpointing and seeding, and a short training script suitable for CPU-based smoke tests.

1. Introduction
---------------
CA15 provides a compact, well-documented codebase to support teaching and rapid experimentation. The included components are sufficient to run small reproducible experiments, verify model shapes and training behavior, and extend to larger experiments (e.g., using environment rollouts and advanced actor-critic algorithms).

2. Files & High-Level API (Detailed)
-----------------------------------
This section lists each top-level file and explains its purpose, public API, and usage notes.

- `src/config.py`
  - Purpose: central `Config` dataclass and YAML loader for experiment hyperparameters.
  - API: `Config()` default instance, `Config.from_dict(d)` to override programmatically, and `Config.from_yaml(path)` to load from a YAML file.
  - Notes: keep defaults minimal and safe for CI (small epochs, CPU device). All configurable fields are annotated in the dataclass.

- `src/data.py`
  - Purpose: `SyntheticDataset` helper for deterministic, fast unit-test data.
  - API: `SyntheticDataset(input_dim, output_dim, size=1024, seed=0)` and method `batches(batch_size)` which yields (states, actions, values) as PyTorch tensors.
  - Notes: Intended only for smoke/prototyping; replace with environment rollouts for real RL experiments.

- `src/model.py`
  - Purpose: `MLPPolicy` (actor producing logits) and `ValueNetwork` (regression head).
  - API: classes are standard torch.nn.Module subclasses. `MLPPolicy.get_action(x, deterministic=False)` returns `(action, log_prob)`.
  - Notes: Small and modular; swapping architectures is straightforward.

- `src/losses.py`
  - Purpose: compact, framework-agnostic implementations of MSE and policy gradient loss.
  - API: `mse_loss(pred, target)`, `policy_gradient_loss(logp, advantage, reduction='mean')`.
  - Notes: Losses are minimal and expect correctly shaped tensors.

- `src/utils.py`
  - Purpose: utilities for seeding, checkpoint save/load and simple atomic file write.
  - API: `set_seed(seed)`, `save_checkpoint(path, model, optimizer, extra=None)`, `load_checkpoint(path, model, optimizer=None)`.
  - Notes: `save_checkpoint` writes atomically via a `.tmp` rename to avoid partial files.

- `src/train.py`
  - Purpose: self-contained `train(cfg, save_path=None)` function that runs a short training loop on `SyntheticDataset`.
  - API: `train(cfg: Config, save_path: Optional[str]=None) -> dict` returns a summary dict with `policy_loss`, `value_loss`, `time_sec`, and `epochs`.
  - CLI: `python -m src.train --cfg configs/default.yaml --save <path>` for quick runs.

- `src/__init__.py`
  - Purpose: convenient exports for top-level usage when importing `src`.

- `configs/default.yaml`
  - Purpose: canonical configuration for demo runs. It sets small default epochs and CPU device for CI.

- `notebooks/quick_demo.ipynb` (template)
  - Purpose: guided, non-executed template showing imports, config loading, dataset inspection, model instantiation, a manual train step, and checkpoint examples.

- `tests/`
  - Purpose: minimal smoke tests (import-safety and forward-pass shape checks). Key tests:
    - `test_ca15_basic.py` — checks Config defaults, forward pass shapes for policy and value.
    - `test_train_import.py` — ensures `train` is importable / callable.

3. Experimental Setup and How-to
--------------------------------
Quick steps to run the demo locally (CPU):

1. Create and activate a Python 3.10+ virtualenv:
   - python -m venv .venv && source .venv/bin/activate
2. Install dependencies (examples):
   - python -m pip install --upgrade pip
   - python -m pip install torch numpy pyyaml matplotlib pytest
3. Run tests:
   - python -m pytest paperAssignments/Assignments1-50/CA15/tests -q
4. Run a short training experiment:
   - python -m src.train --cfg configs/default.yaml
   - or from Python: `from train import train; train(Config(epochs=2))`

4. Experiments, Metrics, and Evaluation
--------------------------------------
- Metrics: report `policy_loss` and `value_loss` as produced by the minimal training loop. For real tasks, include episodic returns, success rates, and standard RL diagnostics (entropy, KL, gradient norms).
- Expected behavior: with synthetic random targets, the goal is stability and deterministic behavior (i.e., runs with identical seeds should match numerically for the same hardware/software environment).
- Evaluation protocol (recommended):
  - Run N seeds (e.g., 5) and report mean ± std for losses and run time.
  - For environment-based experiments, report episode return curves and choose common smoothing windows (e.g., 10- or 100-episode moving average).

5. Reproducibility Details
--------------------------
- Seeding: `utils.set_seed(seed)` sets `random`, `numpy`, and `torch` seeds. For full reproducibility, pin package versions and record `torch.__version__` and system info.
- Checkpointing: use `save_checkpoint`/`load_checkpoint` to persist model/optimizer and optional metadata. Always save config (`cfg.__dict__`) in `extra` for reproducibility.
- Docker/CI: For stricter reproducibility use pinned docker images or record a `requirements.txt` with exact versions.

6. Limitations, Extensions & Notes
---------------------------------
- Limitations: CA15 is intentionally compact and uses synthetic data and simple losses; it is not a production RL system.
- Extensions: add environment rollouts, advantage estimation (GAE), entropy bonuses, network snapshots, multi-environment sampling, and realistic logging (TensorBoard/Weights & Biases).

7. Development Notes & Checklist
-------------------------------
Before merging or publishing:
- [ ] All new code is import-safe and covered by quick smoke tests.
- [ ] All hyperparameters are exposed in `configs/default.yaml`.
- [ ] Notebook templates are non-executed (safe to open) and explain how to run experiments.
- [ ] Save artifacts include config metadata and git commit hash.

8. Contact & Attribution
------------------------
- Maintainers: course repo maintainers (see project root `README.md`)

Appendix — Example Usage
------------------------
Example: small programmatic run

```python
from config import Config
from train import train

cfg = Config(epochs=2, batch_size=32)
res = train(cfg)
print(res)
```

Appendix — File Map
-------------------
- `src/config.py`: Config dataclass and YAML loading
- `src/data.py`: SyntheticDataset
- `src/model.py`: MLPPolicy, ValueNetwork
- `src/losses.py`: mse_loss, policy_gradient_loss
- `src/utils.py`: set_seed, save_checkpoint, load_checkpoint
- `src/train.py`: train function and CLI

Detailed Additions and Examples
------------------------------
- Expected training output (example):

```json
{"policy_loss": 0.1234, "value_loss": 0.5678, "time_sec": 2.34, "epochs": 5}
```

- Config fields (from `src/config.py`):
  - `seed` (int): RNG seed used for reproducibility (default 0)
  - `input_dim` (int): dimensionality of input/state features (default 8)
  - `hidden_dim` (int): hidden layer size (default 64)
  - `output_dim` (int): number of discrete actions (default 4)
  - `lr` (float): learning rate for optimizers (default 1e-3)
  - `device` (str): device string, e.g., `cpu` or `cuda` (default `cpu`)
  - `batch_size` (int): training batch size (default 32)
  - `epochs` (int): number of training epochs (default 10)

Notebook cell map (`notebooks/quick_demo.ipynb`)
- Cell 1 (markdown): Title and overview
- Cell 2 (python): Imports and environment checks
- Cell 3 (python): Load config and override values for quick run
- Cell 4 (python): Instantiate `SyntheticDataset` and visualize samples
- Cell 5 (python): Instantiate `MLPPolicy` and `ValueNetwork` and verify shapes
- Cell 6 (python): Demonstrate losses and metrics computation
- Cell 7 (python): Manual single training step (forward/backward)
- Cell 8 (python): Short training loop example using `train.train` (commented)
- Cell 9 (python): Checkpoint save/load example (commented)
- Cell 10 (python): Evaluation/visualization hints
- Cell 11 (python): Reproducibility/seeding demonstration
- Cell 12 (markdown): Running unit tests and debugging checklist

Testing and Coverage Notes
--------------------------
- Tests are intentionally minimal and focused on import-safety and forward-pass correctness.
- To add tests, follow the patterns in `tests/` and keep them fast (no heavy training loops).
- Recommended tests:
  - Import and instantiation tests for modules
  - Shape checks for model forward passes
  - Checkpoint save/load round-trip (save, load, assert parameter equality)

Troubleshooting & FAQ
---------------------
- If tests fail with import errors, ensure `PYTHONPATH` includes the package `src/` directory (the tests set this up automatically in their top-level files).
- If training crashes with CUDA errors, set `device` to `cpu` in `configs/default.yaml` and re-run.
- If losses are NaN or infinite: check input ranges, lower learning rate, or try gradient clipping.

Contribution Guide & PR Checklist
---------------------------------
- Proposed changes should be small and isolated to a single CA where possible.
- Include a brief description of the change, rationale, and test plan in the PR description.
- Run the CA tests locally before opening a PR.
- PR checklist:
  - [ ] Tests added/updated
  - [ ] README/REPORT updated where relevant
  - [ ] Notebooks added or updated (non-executed)

License and Citation
--------------------
- This CA follows the project-wide license in the repository root (see `LICENSE`). Cite the course or project if used in external work.

References
----------
- Sutton & Barto, "Reinforcement Learning: An Introduction"


