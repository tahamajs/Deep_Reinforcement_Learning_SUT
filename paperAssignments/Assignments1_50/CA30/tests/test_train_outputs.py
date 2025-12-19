import json
from pathlib import Path
from ca30.config import ExperimentConfig
from ca30.utils import set_seed, make_rng
from ca30.model import BaseModel
from ca30.train import train


def test_train_writes_metrics_and_figure(tmp_path: Path):
    cfg = ExperimentConfig(seed=0, input_dim=4, hidden_dim=8, output_dim=3, epochs=1, batch_size=4)
    out_dir = tmp_path / "demo_out"
    set_seed(cfg.seed)
    rng = make_rng(cfg.seed)
    model = BaseModel(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, seed=cfg.seed)
    res = train(cfg, model, rng, max_steps=2, out_dir=out_dir)
    metrics_file = out_dir / "metrics.json"
    fig_file = out_dir / "figure1.png"
    assert metrics_file.exists(), f"metrics file not found: {metrics_file}"
    assert fig_file.exists(), f"figure not found: {fig_file}"
    metrics = json.loads(metrics_file.read_text())
    assert "loss" in metrics and len(metrics["loss"]) == 2
