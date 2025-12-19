import torch


def test_policy_gradient_and_value_loss_shapes():
    from pathlib import Path
    import importlib.util

    base = Path(__file__).resolve().parents[2] / "src"
    spec = importlib.util.spec_from_file_location("ca21.losses", str(base / "losses.py"))
    losses_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses_mod)  # type: ignore

    pg_loss = losses_mod.policy_gradient_loss
    v_loss = losses_mod.value_mse_loss

    batch = 5
    logp = torch.randn(batch)
    adv = torch.randn(batch)
    values = torch.randn(batch)
    targets = torch.randn(batch)

    l1 = pg_loss(logp, adv)
    l2 = v_loss(values, targets)

    assert isinstance(l1, torch.Tensor)
    assert l1.shape == ()
    assert isinstance(l2, torch.Tensor)
    assert l2.shape == ()


def test_losses_check_shape_mismatch_raises():
    from pathlib import Path
    import importlib.util

    base = Path(__file__).resolve().parents[2] / "src"
    spec = importlib.util.spec_from_file_location("ca21.losses", str(base / "losses.py"))
    losses_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses_mod)  # type: ignore

    pg_loss = losses_mod.policy_gradient_loss
    v_loss = losses_mod.value_mse_loss

    import pytest

    with pytest.raises(ValueError):
        pg_loss(torch.randn(3), torch.randn(4))

    with pytest.raises(ValueError):
        v_loss(torch.randn(3), torch.randn(4))
