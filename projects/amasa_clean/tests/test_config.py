from pathlib import Path

from projects.amasa_clean.amasa.core.config import load_config, merge_dicts, apply_named_overlays


def test_config_load_and_merge():
    root = Path(__file__).resolve().parents[1]
    base = root / "configs" / "base.yaml"
    cfg = load_config(base)
    assert "experiment" in cfg
    assert cfg["algo"]["name"] in {"cql", "iql", "sac_lag", "ppo_lag"}

    merged = merge_dicts(cfg, {"algo": {"name": "iql"}, "scenario": {"type": "perturbed"}})
    assert merged["algo"]["name"] == "iql"
    assert merged["scenario"]["type"] == "perturbed"


def test_overlay_order_keeps_preset_runtime():
    root = Path(__file__).resolve().parents[1]
    base = root / "configs" / "base.yaml"
    cfg = load_config(base)
    merged = apply_named_overlays(cfg, root / "configs", algo="cql", scenario="nominal", preset="smoke")
    assert merged["train"]["steps"] == 400
