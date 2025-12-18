import torch
from mamba_core.actor import Actor
from mamba_core.value import ValueNet


def test_actor_value_shapes():
    B = 3
    latent_dim = 32
    morph_dim = 16
    act_dim = 6
    actor = Actor(latent_dim=latent_dim, morph_dim=morph_dim, act_dim=act_dim)
    value = ValueNet(latent_dim=latent_dim, morph_dim=morph_dim)
    z = torch.randn(B, latent_dim)
    z_m = torch.randn(B, morph_dim)
    mu, std = actor(z, z_m)
    assert mu.shape == (B, act_dim)
    assert std.shape == (B, act_dim)
    v = value(z, z_m)
    assert v.shape == (B,)







