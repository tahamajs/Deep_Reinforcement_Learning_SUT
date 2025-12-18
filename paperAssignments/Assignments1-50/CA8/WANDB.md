W&B integration (MaxSink / CA8)

Quick setup

- Install: pip install wandb
- Login once: wandb login

Enable in code

- Open `paperAssignments/Assignments1-50/CA8/src/config.py` and set:

```python
cfg.use_wandb = True
```

Run

- Start your training as usual: python paperAssignments/Assignments1-50/CA8/scripts/train.py
- W&B run will initialize automatically if `wandb` is installed and `cfg.use_wandb` is True.

Notes

- W&B is optional; the trainer falls back to TensorBoard if wandb is not available.
- For reproducibility, include the `cfg.as_dict()` sent to W&B in your run config.

Sweeps

- Example sweep config: `paperAssignments/Assignments1-50/CA8/wandb_sweep.yaml`
- Start a sweep:
  1. wandb sweep paperAssignments/Assignments1-50/CA8/wandb_sweep.yaml
  2. wandb agent <SWEEP_ID>






