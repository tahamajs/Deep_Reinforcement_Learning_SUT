import os
from pathlib import Path


def test_save_and_load_checkpoint(tmp_path):
    from pathlib import Path
    import importlib.util

    base = Path(__file__).resolve().parents[2] / "src"
    spec = importlib.util.spec_from_file_location("ca21.utils", str(base / "utils.py"))
    utils_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_mod)  # type: ignore

    path = tmp_path / "model.ckpt"
    state = {"a": 1}
    utils_mod.save_checkpoint(str(path), state)
    assert path.exists()

    loaded = utils_mod.load_checkpoint(str(path))
    assert loaded["a"] == 1

    # missing file raises
    try:
        utils_mod.load_checkpoint(str(path.with_suffix('.bad')))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
