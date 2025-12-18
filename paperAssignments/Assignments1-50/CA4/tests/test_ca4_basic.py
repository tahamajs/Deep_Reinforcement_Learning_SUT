import pytest

try:
    import torch
except Exception:
    torch = None
    pytest.skip("torch not available — skipping CA4 tests", allow_module_level=True)

from src.losses import quantile_huber_loss, cvar_tail
from src.model import SCASReg, QuantileMLP


def test_quantile_huber_and_grad():
    B, N = 4, 8
    pred = torch.randn(B, N, requires_grad=True)
    target = torch.randn(B, N)
    taus = (torch.arange(N).float() + 0.5) / N
    loss = quantile_huber_loss(pred, target, taus, kappa=1.0)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert pred.grad is not None


def test_cvar_tail_monotonicity():
    qs = torch.linspace(-1.0, 1.0, steps=10).unsqueeze(0).repeat(2, 1)
    # make second row strictly larger
    qs[1] += 0.5
    c1 = cvar_tail(qs[0:1], alpha=0.2)
    c2 = cvar_tail(qs[1:2], alpha=0.2)
    assert (c2 > c1).item()


def test_scas_loss_zero_when_matches():
    s_dim = 5
    a_dim = 2
    scas = SCASReg(s_dim, a_dim)
    s = torch.randn(3, s_dim)
    a = torch.randn(3, a_dim)
    s_next = scas(s, a).detach()  # make next exactly the model's forward
    loss = scas.loss(s, a, s_next)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)














