import torch
from paperAssignments.Assignments1_50.CA19.src.model import ActorCriticEnsemble


def test_forward_shapes():
    obs_dim = 4
    action_dim = 2
    model = ActorCriticEnsemble(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, ensemble_size=3)
    B = 5
    obs = torch.randn(B, obs_dim)
    logits, values = model.forward(obs)
    assert logits.shape == (B, action_dim)
    assert values.shape == (3, B)
    actions, logp, mean_v = model.act(obs)
    assert actions.shape == (B,)
    assert logp.shape == (B,)
    assert mean_v.shape == (B,)
