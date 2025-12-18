from pathlib import Path
from src.config import load_config, save_config, TrainConfig

def test_config_load_and_save(tmp_path: Path):
    cfg = TrainConfig(seed=7, epochs=2, batch_size=4, input_dim=8)
    p = tmp_path / "used_config.yaml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.seed == cfg.seed
    assert loaded.epochs == cfg.epochs
    assert loaded.input_dim == cfg.input_dim
