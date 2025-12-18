import torch
from mamba_core.morph_encoder import MorphEncoder


def test_morph_encoder_shapes():
    B, T, obs_dim, act_dim = 4, 10, 24, 6
    enc = MorphEncoder(obs_dim=obs_dim, act_dim=act_dim, latent_dim=16)
    obs = torch.randn(B, T, obs_dim)
    act = torch.randn(B, T, act_dim)
    rew = torch.randn(B, T)
    done = torch.zeros(B, T)
    z, mu, logvar = enc(obs, act, rew, done)
    assert z.shape == (B, 16)
    assert mu.shape == (B, 16)
    assert logvar.shape == (B, 16)
















