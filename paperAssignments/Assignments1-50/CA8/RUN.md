Running MaxSink (CA8)

Quick start (local)

1. Create a virtualenv and install dependencies:

   python -m venv .venv && source .venv/bin/activate
   pip install -r paperAssignments/Assignments1-50/CA8/requirements.txt

2. Run with default config:

   python paperAssignments/Assignments1-50/CA8/scripts/run.py

3. Run with an example config (Procgen):

   python paperAssignments/Assignments1-50/CA8/scripts/run.py --config paperAssignments/Assignments1-50/CA8/configs/procgen.yaml

Logging
- TensorBoard logs are saved to `cfg.tb_logdir` (default `runs/ca8`). Start TensorBoard:
  tensorboard --logdir runs/ca8
- W&B: set up and login (`pip install wandb` && `wandb login`). W&B is enabled by default in cfg.

W&B sweeps

1. Create the sweep:
   wandb sweep paperAssignments/Assignments1-50/CA8/wandb_sweep.yaml
   Note the returned SWEEP_ID.
2. Start an agent:
   wandb agent <SWEEP_ID>

Or use the provided helper:

   python paperAssignments/Assignments1-50/CA8/scripts/wandb_sweep_agent.py --count 1

Notes
- For MuJoCo/Procgen, ensure the appropriate packages are installed (mujoco, procgen).
- Use `python -m pip install -r paperAssignments/Assignments1-50/CA8/requirements.txt` to install everything.

