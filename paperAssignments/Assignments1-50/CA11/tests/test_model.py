import os
import sys
import torch

# ensure src is importable when running tests from repo root
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from src.model import TWMSSDModel, SSMBlock
from src.tokenizer import SimpleVQVAE
from src.data import RandomTrajectoryDataset


def test_twmssd_forward_shapes():
    B, L, D = 2, 16, 32
    model = TWMSSDModel(d_model=D, n_heads=4, n_layers=2)
    obs = torch.randn(B, L, D)
    acts = torch.randn(B, L, D)
    pred_obs, pred_reward = model(obs, acts)
    assert pred_obs.shape == (B, L, D)
    assert pred_reward.shape == (B, L, 1)


def test_ssmblock_shapes():
    B, L, D = 3, 10, 24
    ssm = SSMBlock(D)
    x = torch.randn(B, L, D)
    out = ssm(x)
    assert out.shape == (B, L, D)


def test_vqvae_tokenizer_roundtrip():
    B, L, D = 2, 8, 16
    vq = SimpleVQVAE(codebook_size=64, d_model=D)
    x = torch.randn(B, L, D)
    recon, indices = vq(x)
    assert recon.shape == x.shape
    assert indices.shape == (B, L)
    decoded = vq.decode_codes(indices)
    assert decoded.shape == (B, L, D)












