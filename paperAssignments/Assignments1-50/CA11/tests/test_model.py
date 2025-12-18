import os
import sys
import torch

# ensure src is importable when running tests from repo root
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from src.model import TWMSSDModel, SSMBlock, LinearAttention
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


def test_linear_attn_equiv_ssm():
    """Verify that the LinearAttention module matches a manual linear-attention computation."""
    torch.manual_seed(0)
    B, L, D = 2, 5, 16
    attn = LinearAttention(d_model=D, n_heads=4)
    x = torch.randn(B, L, D)
    out1 = attn(x)

    # Manual computation using the same qkv projections and feature map
    qkv = attn.qkv(x)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = q.view(B, L, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = k.view(B, L, attn.n_heads, attn.head_dim).transpose(1, 2)
    v = v.view(B, L, attn.n_heads, attn.head_dim).transpose(1, 2)
    qf = attn.feature_map(q)
    kf = attn.feature_map(k)
    KV_sum = torch.einsum("bhld,bhlv->bhdv", kf, v)
    out = torch.einsum("bhld,bhdv->bhlv", qf, KV_sum)
    out = out.transpose(1, 2).contiguous().view(B, L, D)
    out2 = attn.out(out)
    assert torch.allclose(out1, out2, atol=1e-6)















