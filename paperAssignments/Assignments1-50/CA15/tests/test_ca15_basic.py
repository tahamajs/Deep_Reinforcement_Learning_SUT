import sys
import pathlib
import torch

# Ensure CA15/src is importable when running tests from repo root
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "CA15" / "src"
sys.path.insert(0, str(SRC))

from config import Config
from model import MLPPolicy, ValueNetwork
from utils import set_seed


def test_import_and_forward():
    cfg = Config()
    set_seed(cfg.seed)
    policy = MLPPolicy(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
    value = ValueNetwork(cfg.input_dim, cfg.hidden_dim)
    x = torch.randn(2, cfg.input_dim)
    logits = policy(x)
    assert logits.shape == (2, cfg.output_dim)
    v = value(x)
    assert v.shape == (2,)













