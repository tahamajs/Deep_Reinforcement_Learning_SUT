import os
import tempfile
import torch
import numpy as np
from src.config import ExperimentConfig
from src.utils import set_seed, save_checkpoint, load_checkpoint, update_lagrange
from src.data import SyntheticDataset
from src.losses import LagrangianLoss


def test_yaml_and_dataclass(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 7\nobs_dim: 3\naction_dim: 2\n")
    cfg = ExperimentConfig.from_yaml(str(p))
    assert cfg.seed == 7
    assert cfg.obs_dim == 3


def test_seed_determinism():
    set_seed(0)
    a = np.random.rand(5)
    set_seed(0)
    b = np.random.rand(5)
    assert np.allclose(a, b)
    # also check PyTorch RNG reproducibility on CPU
    torch.manual_seed(0)
    t1 = torch.randn(4)
    torch.manual_seed(0)
    t2 = torch.randn(4)
    assert torch.allclose(t1, t2)


def test_checkpoint_roundtrip(tmp_path):
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(3, 2)

    d = Dummy()
    opt = torch.optim.Adam(d.parameters(), lr=1e-3)
    ck = tmp_path / "ckpt.pt"
    save_checkpoint(str(ck), d, opt, extra={"a": 1})
    data = load_checkpoint(str(ck), d, opt)
    assert "model_state" in data
    assert data.get("extra", {}).get("a") == 1


def test_update_lagrange_and_lagrange_loss():
    mu = 0.0
    new_mu = update_lagrange(mu, constraint_value=1.0, c=0.5, lr=1.0, max_mu=10.0)
    assert new_mu >= 0.5
    lag = LagrangianLoss(mu=new_mu, constraint_threshold=0.5)
    pg = torch.tensor(1.0)
    cons = torch.tensor([0.6, 0.7])
    combined = lag(pg, cons)
    assert combined.shape == ()


def test_synthetic_dataset_shapes():
    ds = SyntheticDataset(num_episodes=5, obs_dim=4, horizon=6, seed=0)
    s, a, r, c = ds.sample_batch(batch_size=3)
    assert s.shape[1] == 4
    assert a.shape[0] == s.shape[0]
    assert r.shape[0] == s.shape[0]
    assert c.shape[0] == s.shape[0]
