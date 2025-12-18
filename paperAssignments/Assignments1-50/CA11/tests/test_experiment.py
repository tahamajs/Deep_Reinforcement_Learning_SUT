import os
import sys
import torch

# make src importable
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from src.config import get_default_config
from src.scripts.experiment import run


def test_experiment_resume(tmp_path):
    cfg = get_default_config()
    # ensure small quick run
    cfg.batch_size = 2
    cfg.seq_len = 8
    cfg.d_model = 32
    cfg.n_heads = 4
    cfg.n_layers = 1
    cfg.lr = 1e-3
    # run a short experiment and get checkpoint path
    save_dir = str(tmp_path / "run1")
    os.makedirs(save_dir, exist_ok=True)
    path = run(cfg, steps=5, save_dir=save_dir)
    assert os.path.exists(path)
    # now resume from that checkpoint
    cfg.resume_ckpt = path
    save_dir2 = str(tmp_path / "run2")
    os.makedirs(save_dir2, exist_ok=True)
    path2 = run(cfg, steps=2, save_dir=save_dir2)
    assert os.path.exists(path2)











