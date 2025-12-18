import math

import torch

from advantage import compute_advantages, update_gamma, update_gamma_with_ema


def test_compute_advantages_simple():
    # deterministic rewards, zero values -> advantages equal discounted sums
    rewards = torch.tensor([1.0, 0.0, 0.0])
    values = torch.tensor([0.0, 0.0, 0.0, 0.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    gamma = 0.9
    lam = 1.0
    adv, returns = compute_advantages(rewards, values, dones, gamma, lam)
    # A0 = 1 + 0 + 0 = 1, A1 = 0, A2 = 0
    assert torch.isclose(adv[0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(adv[1], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(adv[2], torch.tensor(0.0), atol=1e-6)


def test_update_gamma_direction_and_clamp():
    g = 0.95
    sigma_target = 1.0
    alpha = 0.01
    gmin = 0.9
    gmax = 0.99
    # if varA < target -> gamma increases
    g_new = update_gamma(g, varA=0.5, sigma_target=sigma_target, alpha=alpha, gmin=gmin, gmax=gmax)
    assert g_new > g
    # if varA > target -> gamma decreases
    g_new2 = update_gamma(g, varA=2.0, sigma_target=sigma_target, alpha=alpha, gmin=gmin, gmax=gmax)
    assert g_new2 < g
    # clamping
    g_high = update_gamma(g, varA=-1000.0, sigma_target=sigma_target, alpha=alpha, gmin=gmin, gmax=gmax)
    assert g_high <= gmax + 1e-12


def test_update_gamma_with_ema_behavior():
    g = 0.95
    ema = None
    alpha = 0.01
    beta = 0.9
    sigma_target = 1.0
    gmin = 0.9
    gmax = 0.99
    g, ema = update_gamma_with_ema(g, varA=0.5, ema_varA=None, alpha=alpha, beta=beta, sigma_target=sigma_target, gmin=gmin, gmax=gmax)
    assert ema == 0.5
    g2, ema2 = update_gamma_with_ema(g, varA=0.6, ema_varA=ema, alpha=alpha, beta=beta, sigma_target=sigma_target, gmin=gmin, gmax=gmax)
    # ema should be between 0.5 and 0.6
    assert 0.5 <= ema2 <= 0.6

