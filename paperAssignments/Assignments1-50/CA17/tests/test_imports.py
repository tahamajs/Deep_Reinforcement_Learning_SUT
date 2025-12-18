def test_imports_and_forward():
    """Simple smoke test: load `src/` modules and run forward on dummy input."""
    import importlib.util
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent / "src"

    spec = importlib.util.spec_from_file_location("ca17.model", str(base / "model.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    MLPPolicy = mod.MLPPolicy

    import torch

    model = MLPPolicy(input_dim=4, output_dim=2, hidden_size=16)
    x = torch.randn(3, 4)
    logits = model(x)
    assert logits.shape == (3, 2)













