# CA16 — Simple RL Policy & Value Module

This folder implements a small, import-safe PyTorch package for Assignment CA16.

Files added:
- `src/config.py` — configuration dataclass and getter
- `src/model.py` — `MLPPolicy` and `MLPValue` models
- `src/losses.py` — policy gradient and value losses
- `src/data.py` — lightweight replay buffer
- `src/utils.py` — seeding and small helpers
- `tests/test_model.py` — unit tests for forward passes and losses

Notes:
- Per repository conventions, modules are import-safe and contain no training loops on import.
- The tests are lightweight and meant as smoke tests; training code belongs in notebooks or scripts outside `src/`.
