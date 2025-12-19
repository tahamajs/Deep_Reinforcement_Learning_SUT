# CA16 — Report

## Abstract
This short project provides small, import-safe PyTorch modules for a discrete-action
policy and a state-value function, along with supporting utilities and a simple
replay buffer. The intent is educational: provide compact reference implementations
for policy-gradient style algorithms and to support quick experiments in notebooks.

## Implementation
- Models:
  - `MLPPolicy(obs_dim, action_dim, hidden_dim)` produces raw action logits and
    exposes `action_distribution` and `get_action` helpers.
  - `MLPValue(obs_dim, hidden_dim)` predicts state values (scalar per observation).
- Losses:
  - `policy_loss(log_probs, advantages, entropy_coeff=0., entropies=None)` — standard
    policy gradient objective (minimize negative expected advantage-weighted log-prob)
    with optional entropy bonus.
  - `value_loss(values, targets)` — mean squared error.
- Data utilities:
  - `ReplayBuffer(capacity)` — in-memory FIFO buffer with `add`, `sample`, and `clear`.
- Misc:
  - `set_seed`, `to_tensor`, `count_parameters`

## Design choices and rationale
- Keep modules import-safe and minimal so they can be used in notebooks without side-effects.
- Use simple MLPs (configurable depth) as they are effective for small RL benchmarks such
  as CartPole.
- Prefer explicit, typed APIs and small unit tests for quick verification.

## How to use
1. Install project dependencies (PyTorch, numpy)
2. Import modules from the package and create models:

```python
from CA16.src.model import MLPPolicy, MLPValue
policy = MLPPolicy(obs_dim=4, action_dim=2)
value = MLPValue(obs_dim=4)
```

3. Compute losses using `policy_loss` and `value_loss`.

## Tests
- Unit tests are included under `tests/` and verify forward passes, loss shapes,
  replay buffer sampling, and small utilities. Run with `pytest` from the repository root.

## Suggested experiments
- Evaluate policy gradient training on `CartPole-v1` using the built-in MLPPolicy
  and MLPValue. Vary `hidden_dim` and learning rate to observe stability.
- Add entropy coefficient sweeping and compare learning curves.
- Compare with a baseline critic-only (value-only) model to measure advantage of
  including a learned value function for variance reduction.

## Evaluation & Reporting
When running experiments, record the following:
- environment name and seed
- network sizes and optimizer hyperparameters
- learning curves (episode return vs training steps)
- average return over 10 evaluation episodes at checkpoints

A recommended structure for a result table:

| Setting | Hidden dim | LR | Entropy coeff | Avg return (final) |
|---|---:|---:|---:|---:|
| CartPole | 128 | 3e-4 | 0.0 | 195.2 |


## Limitations and future work
- No continuous-action policy implemented (Gaussian actor would be a natural addition).
- No built-in trainer or experiment harness; left to notebooks to keep the package
  import-safe and focused.

## References
- Sutton & Barto, Reinforcement Learning: An Introduction
- OpenAI Gym (environments such as CartPole)
