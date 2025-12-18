Usage and Math Addendum

This file supplements the existing `README.md` for CA3 with concise usage instructions and the math-to-code mapping.

Usage

1. Create a virtual environment and install dependencies (PyTorch, Gym, Matplotlib):

   python -m venv .venv && source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch gymnasium matplotlib

2. Run the notebook `notebooks/demo.ipynb` (open in Jupyter). The notebook imports the `src` package and runs a REINFORCE training loop.

Math-to-code mapping

- Objective and gradient: `report.tex` (Methodology) and `src/losses.py::reinforce_loss`.
- Policy network: `src/model.py::MLPPolicy` implements $\pi_\theta(a|s)$; sampling and log-probabilities are provided by `get_action`.
- Episode collection: `src/data.py::collect_episode`.
- Returns computation: `src/utils.py::discounted_returns` and `returns_to_tensor`.

Notes

- Hyperparameters: edit `src/config.py::Config`.
- Checkpoints: use `src/utils.py::save_checkpoint` / `load_checkpoint`.











