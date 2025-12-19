from src.config import load_config, Config


def test_load_config(tmp_path):
    cfg_text = """
seed: 123
learning_rate: 0.01
batch_size: 16
"""
    p = tmp_path / "temp_config.yaml"
    p.write_text(cfg_text)
    cfg = load_config(str(p))
    assert isinstance(cfg, Config)
    assert cfg.seed == 123
    assert cfg.learning_rate == 0.01
    assert cfg.batch_size == 16
