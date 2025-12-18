import torch
from model import ActorCritic


def test_actor_critic_forward_shapes():
    obs_dim = 4
    action_dim = 2
    net = ActorCritic(obs_dim, action_dim, hidden_sizes=(32, 32))
    batch = torch.randn(5, obs_dim)
    actions, logp = net.act(batch)
    assert actions.shape == (batch.shape[0],)
    assert logp.shape == (batch.shape[0],)
    v = net.get_value(batch)
    assert v.shape == (batch.shape[0],)


def test_evaluate_actions_shapes():
    obs = torch.randn(6, obs_dim := 4)
    actions = torch.randint(0, 2, (6,))
    net = ActorCritic(obs_dim, action_dim:=2, hidden_sizes=(16,))
    logp, entropy, value = net.evaluate_actions(obs, actions)
    assert logp.shape == (6,)
    assert entropy.shape == (6,)
    assert value.shape == (6,)
