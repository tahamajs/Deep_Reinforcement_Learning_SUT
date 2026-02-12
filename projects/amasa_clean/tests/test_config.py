from pathlib import Path

from projects.amasa_clean.amasa.core.config import load_config, merge_dicts


def test_config_load_and_merge():
    root = Path(__file__).resolve().parents[1]
    base = root / "configs" / "base.yaml"
    cfg = load_config(base)
    assert "experiment" in cfg
    assert cfg["algo"]["name"] in {"cql", "iql", "sac_lag", "ppo_lag"}

    merged = merge_dicts(cfg, {"algo": {"name": "iql"}, "scenario": {"type": "perturbed"}})
    assert merged["algo"]["name"] == "iql"
    assert merged["scenario"]["type"] == "perturbed"
