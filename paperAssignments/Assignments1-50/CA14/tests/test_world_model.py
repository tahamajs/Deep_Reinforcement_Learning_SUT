import torch
from mamba_core.world_model import WorldModel


def test_world_model_forward():
    B = 2
    obs_dim = 24
    act_dim = 6
    wm = WorldModel(
        obs_dim=obs_dim, act_dim=act_dim, deter_dim=64, stoch_dim=32, morph_dim=16
    )
    device = torch.device("cpu")
    h, z = wm.init_state(B, device)
    a = torch.randn(B, act_dim)
    z_m = torch.randn(B, 16)
    obs = torch.randn(B, obs_dim)
    h_new, z_new, mu_post, logvar_post = wm.observe(h, a, z_m, obs)
    assert h_new.shape[0] == B
    assert z_new.shape == (B, 32)
    recon = wm.decode_obs(h_new, z_new, z_m)
    assert recon.shape == (B, obs_dim)
    r = wm.predict_reward(h_new, z_new, z_m)
    assert r.shape == (B,)
    d = wm.predict_discount(h_new, z_new, z_m)
    assert d.shape == (B,)












