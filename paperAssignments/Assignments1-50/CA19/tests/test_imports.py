import importlib.util
from pathlib import Path

base = Path(__file__).resolve().parent.parent / "src"


def _load(fname: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(base / fname))
    m = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(m)
    return m


def test_imports():
    _load("config.py", "ca19.config")
    _load("model.py", "ca19.model")
    _load("losses.py", "ca19.losses")
    _load("utils.py", "ca19.utils")
    _load("data.py", "ca19.data")
    assert True






