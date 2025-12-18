import torch
from src.crosshq.model import CrossQCritic, GaussianPolicy
from src.crosshq.mcts_adapter import CrossHQMCTSAdapter
from src.mcts.puct import PUCT


def test_crosshq_adapter_runs():
    s_dim = 4
    a_dim = 2
    critic = CrossQCritic(state_dim=s_dim, action_dim=a_dim, hidden_dim=64, depth=2)
    actor = GaussianPolicy(obs_dim=s_dim, action_dim=a_dim, hidden=64, depth=2)

    # define a small discrete action set (e.g., 3 candidate continuous actions)
    action_set = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.5]),
    ]

    adapter = CrossHQMCTSAdapter(critic, actor, action_set)
    # should return 3 priors summing to ~1
    priors = adapter.policy(torch.randn(s_dim))
    assert len(priors) == len(action_set)
    assert abs(sum(priors) - 1.0) < 1e-4

    # value should be finite
    v = adapter.value(torch.randn(s_dim))
    assert isinstance(v, float)

    # Use adapter in PUCT
    puct = PUCT(adapter, action_space=[0, 1, 2], c_puct=1.0)
    root = puct.search(0, num_simulations=10)
    visits = sum(child.visits for child in root.children.values())
    assert visits == 10












