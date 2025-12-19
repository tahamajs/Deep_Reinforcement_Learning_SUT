import tempfile
import os
import yaml

from config_loader import load_config_from_yaml
from config import cfg


def test_load_config_updates_cfg(tmp_path):
    # create a temporary yaml with a known key and an unknown key
    p = tmp_path / "tmp_cfg.yaml"
    data = {"beta": 0.123, "some_unknown": "value"}
    p.write_text(yaml.safe_dump(data))

    # load config
    loaded = load_config_from_yaml(str(p))
    assert isinstance(loaded, dict)
    # known key updated
    assert abs(cfg.beta - 0.123) < 1e-6
    # unknown key did not create new attribute on cfg
    assert not hasattr(cfg, "some_unknown")
