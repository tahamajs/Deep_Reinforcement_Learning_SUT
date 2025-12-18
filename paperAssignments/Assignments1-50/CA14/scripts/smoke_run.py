"""Minimal smoke run to instantiate models and run a forward pass."""

import torch
from mamba_core.morph_encoder import MorphEncoder
from mamba_core.world_model import WorldModel
from mamba_core.actor import Actor
from mamba_core.value import ValueNet


def smoke():
    B, T = 2, 8
    obs_dim, act_dim = 24, 6
    morph_dim = 16
    wm = WorldModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        deter_dim=64,
        stoch_dim=32,
        morph_dim=morph_dim,
    )
    morph = MorphEncoder(obs_dim=obs_dim, act_dim=act_dim, latent_dim=morph_dim)
    actor = Actor(latent_dim=32, morph_dim=morph_dim, act_dim=act_dim)
    value = ValueNet(latent_dim=32, morph_dim=morph_dim)

    obs = torch.randn(B, T, obs_dim)
    act = torch.randn(B, T, act_dim)
    rew = torch.randn(B, T)
    done = torch.zeros(B, T)

    z_m, mu_m, logvar_m = morph(obs, act, rew, done)
    h, z = wm.init_state(B, device=torch.device("cpu"))
    a = torch.randn(B, act_dim)
    h, z, mu_post, logvar_post = wm.observe(h, a, z_m, obs[:, 0])
    recon = wm.decode_obs(h, z, z_m)
    r = wm.predict_reward(h, z, z_m)
    print("smoke run shapes:", recon.shape, r.shape)


if __name__ == "__main__":
    smoke()







