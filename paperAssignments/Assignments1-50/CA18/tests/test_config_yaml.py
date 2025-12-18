from pathlib import Path
from config import Config


def test_load_yaml_debug_config(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    cfg = Config.load_yaml(root / "configs" / "debug.yaml")
    assert isinstance(cfg, Config)
    assert isinstance(cfg.hidden_sizes, tuple)
    assert all(isinstance(x, int) for x in cfg.hidden_sizes)
    assert cfg.epochs == 3
