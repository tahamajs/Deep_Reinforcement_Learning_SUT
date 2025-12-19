import pytest

pytest.importorskip("torch")


def test_imports():
    import importlib
    modules = ["src.config", "src.utils", "src.model", "src.losses", "src.data"]
    for m in modules:
        mod = importlib.import_module(m)
        assert mod is not None
