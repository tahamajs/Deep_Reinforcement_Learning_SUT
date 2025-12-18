def test_imports_and_forward():
    """Simple smoke test: import package modules and run forward on dummy input."""
    import torch

    from paperAssignments.Assignments1_50.CA17.src.model import MLPPolicy  # type: ignore

    model = MLPPolicy(input_dim=4, output_dim=2, hidden_size=16)
    x = torch.randn(3, 4)
    logits = model(x)
    assert logits.shape == (3, 2)

