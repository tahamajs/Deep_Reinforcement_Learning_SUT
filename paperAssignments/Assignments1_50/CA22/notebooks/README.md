# Notebooks

This folder contains demo notebooks to run small reproducible experiments using the CA22 scaffold.

- `demo.ipynb` is a lightweight example that demonstrates loading a config (`configs/debug.yaml`), seeding, building `PolicyNet` and `ValueNet`, running a few synthetic episodes, and saving a couple of debug figures to `outputs/`.

Notes:
- Notebooks are not executed in the repository; please run them locally in a conda/venv environment with `pip install -r requirements.txt`.
- Prefer running with the `debug.yaml` config for quick runs.
