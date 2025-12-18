# Notebooks

This folder can hold short exploratory notebooks that demonstrate how to run experiments and visualize results. Prefer keeping notebooks lightweight and use them for plotting or analysis only; keep reproducible training logic in `train.py` and `src/`.

Suggested notebook contents:
- `01-experiment.ipynb` — small example that loads `used_config.yaml`, displays training curves, and shows model predictions on a few synthetic examples.

Note: Do not put heavyweight runs in notebooks; use `train.py` for reproducibility and artifact saving.
