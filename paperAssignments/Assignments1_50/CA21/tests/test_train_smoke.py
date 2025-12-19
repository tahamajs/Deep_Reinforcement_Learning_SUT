def test_train_smoke_runs_quickly():
    """Smoke test: run a very short training loop and assert metrics are returned."""
    from pathlib import Path
    import importlib.util

    base = Path(__file__).resolve().parents[2] / "src"
    spec = importlib.util.spec_from_file_location("ca21.train", str(base / "train.py"))
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)  # type: ignore

    # create a tiny config-like object using the Config dataclass
    spec_cfg = importlib.util.spec_from_file_location("ca21.config", str(base / "config.py"))
    cfg_mod = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(cfg_mod)  # type: ignore

    cfg = cfg_mod.Config(seed=42, input_dim=8, hidden_dim=16, action_dim=4, lr=1e-3, batch_size=4, epochs=1)
    metrics = train_mod.train(cfg=cfg, num_samples=32, checkpoint_path=None)

    assert isinstance(metrics, dict)
    assert "final_pg_loss" in metrics
    assert "final_value_loss" in metrics
    assert "seconds" in metrics
