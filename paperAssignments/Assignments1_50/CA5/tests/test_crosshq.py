import torch
from src.crosshq.model import CrossQCritic, GaussianPolicy
from src.crosshq.losses import CrossHQLoss
from src.crosshq.relabel import off_policy_correction


def test_critic_and_actor_shapes():
    torch.manual_seed(0)
    B = 4
    s_dim = 8
    a_dim = 3
    critic = CrossQCritic(s_dim, a_dim, hidden_dim=64, depth=2, bn_momentum=0.01)
    actor = GaussianPolicy(s_dim, a_dim, hidden=64, depth=2)

    obs = torch.randn(B, s_dim)
    act = torch.randn(B, a_dim)

    # forward critic on concatenated input
    x = torch.cat([obs, act], dim=-1)
    q1, q2 = critic.forward(x)
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)

    # actor outputs
    a_sample, logp = actor.rsample_and_logprob(obs)
    assert a_sample.shape == (B, a_dim)
    assert logp.shape == (B, 1)


def test_crosshq_loss_computes():
    torch.manual_seed(1)
    B = 6
    s_dim = 10
    a_dim = 2
    critic = CrossQCritic(s_dim, a_dim, hidden_dim=64, depth=2)
    actor = GaussianPolicy(s_dim, a_dim, hidden=64, depth=2)
    loss_module = CrossHQLoss(critic, actor, gamma=0.99, alpha=0.0)

    obs = torch.randn(B, s_dim)
    action = torch.randn(B, a_dim)
    reward = torch.randn(B, 1)
    next_obs = torch.randn(B, s_dim)
    mask = torch.ones(B, 1)

    loss = loss_module(obs, action, reward, next_obs, mask)
    assert torch.isfinite(loss).all()
    assert loss.dim() == 0


def test_off_policy_correction_shape_and_device():
    torch.manual_seed(2)
    B = 3
    s_dim = 5
    a_dim = 2
    c = 4
    manager_batch = {
        "obs": torch.randn(B, s_dim),
        "action_seq": torch.randn(B, c, a_dim),
        "goal": torch.randn(B, a_dim),
        "next_obs": torch.randn(B, s_dim),
    }

    # create a toy worker policy with dist(obs_goal) method
    class ToyPolicy:
        def dist(self, x):
            # return Normal with small std to make log_prob finite
            mu = x[..., :a_dim]
            std = torch.ones_like(mu) * 0.5
            return torch.distributions.Normal(mu, std)

    toy = ToyPolicy()
    relabeled = off_policy_correction(manager_batch, toy, k=3)
    assert relabeled.shape == (B, a_dim)
















