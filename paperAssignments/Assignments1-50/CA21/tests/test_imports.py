def test_imports():
    """Smoke test: ensure modules import without side effects using file-based loader.

    Many folders in this repository include characters that prevent normal package imports
    (e.g., hyphens). Load modules directly from their file paths.
    """
    import importlib.util
    from pathlib import Path

    base = Path(__file__).resolve().parents[2] / "src"
    assert base.exists()

    files = ["config.py", "model.py", "losses.py", "data.py", "utils.py", "__init__.py"]
    for fn in files:
        path = base / fn
        spec = importlib.util.spec_from_file_location(f"ca21.{fn}", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        assert module is not None









