def test_package_imports():
    import importlib

    m = importlib.import_module("ca30")
    assert hasattr(m, "__version__")
