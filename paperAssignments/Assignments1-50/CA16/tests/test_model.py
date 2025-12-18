import torch
from paperAssignments.Assignments1_50.CA16.src.model import MLPPolicy, MLPValue
from paperAssignments.Assignments1_50.CA16.src.losses import policy_loss, value_loss


def test_policy_and_value_forward():
    obs_dim = 4
    action_dim = 2
    batch = 8
    obs = torch.randn(batch, obs_dim)

    policy = MLPPolicy(obs_dim, action_dim, hidden_dim=32)
    value = MLPValue(obs_dim, hidden_dim=32)

    logits = policy(obs)
    assert logits.shape == (batch, action_dim)

    actions, logp = policy.get_action(obs)
    assert actions.shape[0] == batch
    assert logp.shape[0] == batch

    vals = value(obs)
    assert vals.shape == (batch,)

    # synthetic advantages/targets
    advantages = torch.randn(batch)
    pg_loss = policy_loss(logp, advantages)
    assert pg_loss.shape == ()

    targets = torch.randn(batch)
    v_loss = value_loss(vals, targets)
    assert v_loss.shape == ()














