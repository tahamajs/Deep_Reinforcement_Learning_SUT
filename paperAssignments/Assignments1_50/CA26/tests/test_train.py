from pathlib import Path

import torch

from src.config import ExperimentConfig, ModelConfig, TrainConfig
from scripts.run_experiment import fit


def test_fit_creates_artifacts(tmp_path: Path):
    model_cfg = ModelConfig(input_dim=1, hidden_dims=(16,), output_dim=1)
    train_cfg = TrainConfig(seed=0, batch_size=16, lr=1e-3, epochs=1, device="cpu")
    cfg = ExperimentConfig(name="test", model=model_cfg, train=train_cfg)
    out = tmp_path / "outputs"
    res = fit(cfg, out)
    # FitResult should have losses list of length epochs
    assert len(res.losses) == 1
    # model file should be saved
    model_path = out / "model.pt"
    assert model_path.exists()
    # state dict should be a dict-like object
    assert isinstance(res.final_state, dict)
    assert "model_state_dict" in res.final_state
    # attempt to load to ensure shape matches
    st = torch.load(model_path)
    assert "0.weight" in st
