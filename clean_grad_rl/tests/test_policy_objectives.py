import torch


def test_ppo_clipped_surrogate_not_exceed_unclipped_for_positive_advantage():
    ratio = torch.tensor([1.5, 1.2, 0.8])
    adv = torch.tensor([1.0, 0.5, 0.7])
    eps = 0.2
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    ppo_term = torch.min(unclipped, clipped)
    assert torch.all(ppo_term <= unclipped + 1e-7)


def test_trpo_kl_is_non_negative():
    p = torch.distributions.Categorical(logits=torch.tensor([[1.0, 0.0, -1.0]]))
    q = torch.distributions.Categorical(logits=torch.tensor([[0.5, 0.2, -0.7]]))
    kl = torch.distributions.kl_divergence(p, q)
    assert torch.all(kl >= 0)
