# CA16 — Simple RL Policy & Value Module ✅

A compact, import-safe PyTorch package intended as a small reference
implementation for policy-gradient style algorithms (discrete actions).

## Contents 🔧

- `src/config.py` — `Config` dataclass and helper `get_default_config()`
- `src/model.py` — `MLPPolicy` and `MLPValue` models
- `src/losses.py` — `policy_loss` and `value_loss`
- `src/data.py` — `ReplayBuffer` in-memory buffer with `add`, `sample`, and `clear`
- `src/utils.py` — `set_seed`, `to_tensor`, `count_parameters`
- `tests/` — unit tests covering forward passes, losses, replay buffer and utils
- `REPORT.md` — short project report and recommended experiments

## Quickstart 💡

1. Install dependencies (tested with PyTorch + numpy):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch numpy pytest
```

2. Run tests from the repository root:

```bash
pytest paperAssignments/Assignments1-50/CA16/tests
```

3. Example usage in a notebook:

```python
from CA16.src.model import MLPPolicy, MLPValue
from CA16.src.losses import policy_loss, value_loss

policy = MLPPolicy(obs_dim=4, action_dim=2)
value = MLPValue(obs_dim=4)

obs = torch.randn(8, 4)
logits = policy(obs)
actions, logp = policy.get_action(obs)
vals = value(obs)
```

## Testing & Development 🔬

- Unit tests are small and deterministic where possible — primarily smoke tests to
ensure shapes and basic behaviors remain correct. Keep changes focused and add
new tests for any substantial feature.

## Implementation notes ⚙️

- The package is kept intentionally minimal and import-safe (no side effects on import).
- `policy_loss` implements the typical advantage-weighted PG objective and
  optionally accepts entropy terms for an entropy bonus.
- `ReplayBuffer.sample` raises a helpful `ValueError` when the requested batch is
  larger than the buffer size.

## Report
See `REPORT.md` for a summary of design choices, suggested experiments, and
reporting templates.

---

If you'd like, I can also add a `Makefile` or `pyproject.toml` for easier dev flow.















