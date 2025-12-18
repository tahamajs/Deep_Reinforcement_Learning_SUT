import torch


def test_model_forward_shapes():
    """Quick forward-pass shape checks for policy and value networks."""
    from pathlib import Path
    import importlib.util

    base = Path(__file__).resolve().parents[2] / "src"
    spec = importlib.util.spec_from_file_location("ca21.model", str(base / "model.py"))
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)  # type: ignore

    Policy = model_mod.MLPPolicy
    Value = model_mod.MLPValue

    batch = 7
    input_dim = 8
    action_dim = 4
    x = torch.randn(batch, input_dim)

    policy = Policy(input_dim=input_dim, hidden_dim=16, action_dim=action_dim)
    value = Value(input_dim=input_dim, hidden_dim=16)

    logits = policy(x)
    assert logits.shape == (batch, action_dim)

    actions = policy.get_action(x)
    assert actions.shape == (batch,)

    values = value(x)
    assert values.shape == (batch,)


