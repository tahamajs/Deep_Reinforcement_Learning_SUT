import torch

from paperAssignments.Assignments1_50.CA17.src.model import MLPPolicy


def test_act_deterministic():
    model = MLPPolicy(input_dim=4, output_dim=2, hidden_size=16)
    x = torch.randn(4)
    a = model.act(x, deterministic=True)
    assert isinstance(a.item(), int)


def test_get_action_dist_batch():
    model = MLPPolicy(input_dim=4, output_dim=3, hidden_size=16)
    x = torch.randn(5, 4)
    dist = model.get_action_dist(x)
    s = dist.sample()
    assert s.shape == (5,)
