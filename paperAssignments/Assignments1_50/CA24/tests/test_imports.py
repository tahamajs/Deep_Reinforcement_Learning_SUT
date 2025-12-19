def test_imports():
    # Simple import smoke test to ensure package is import-safe
    import importlib
    import src  # noqa: F401
    importlib.reload(src)

    from src.config import Config  # noqa: F401
    from src.model import SimpleMLP  # noqa: F401
    from src.data import SyntheticRegressionDataset  # noqa: F401
    from src.experiment import run_experiment  # noqa: F401
    assert Config is not None
    assert SimpleMLP is not None
    assert SyntheticRegressionDataset is not None
    assert run_experiment is not None


def test_run_experiment_smoke():
    # Run the experiment with a tiny config to ensure no runtime errors on CPU
    from src.config import Config
    from src.experiment import run_experiment

    cfg = Config(epochs=1, batch_size=16)
    out = run_experiment(cfg)
    assert isinstance(out, dict)
    assert "final_train_loss" in out
    assert "config" in out


def test_experiment_reproducibility():
    # Check reproducibility across repeated runs with same seed
    from src.config import Config
    from src.experiment import run_experiment

    cfg1 = Config(epochs=1, batch_size=16, seed=123)
    cfg2 = Config(epochs=1, batch_size=16, seed=123)
    out1 = run_experiment(cfg1)
    out2 = run_experiment(cfg2)
    # allow tiny numerical differences (float equality is brittle)
    assert abs(out1["final_train_loss"] - out2["final_train_loss"]) < 1e-8
