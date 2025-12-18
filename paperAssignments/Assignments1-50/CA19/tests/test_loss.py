import torch
import importlib.util
from pathlib import Path

base = Path(__file__).resolve().parent.parent / "src"

spec = importlib.util.spec_from_file_location("ca19.losses", str(base / "losses.py"))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)
value_ensemble_variance = mod.value_ensemble_variance
actor_loss = mod.actor_loss
critic_loss = mod.critic_loss


def test_losses_finite():
    B = 6
    A = 3
    logits = torch.randn(B, A)
    actions = torch.randint(0, A, (B,))
    advantages = torch.randn(B)
    values = torch.randn(3, B)
    targets = torch.randn(B)
    var = value_ensemble_variance(values)
    a_loss = actor_loss(logits, actions, advantages, var, beta=0.1)
    c_loss = critic_loss(values, targets)
    assert torch.isfinite(a_loss)
    assert torch.isfinite(c_loss)


