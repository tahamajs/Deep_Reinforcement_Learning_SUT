import torch
from planner.advanced import simulate_branches_advanced
from planner.distill import branch_to_action_distribution, select_branch_action


class StubRSSM:
    def step(self, z, a):
        z_next = z + 0.01
        return z_next, 0.5, 1.0


class StubActor:
    def sample(self, z):
        return torch.zeros((z.shape[0], 1), device=z.device)


def test_advanced_simulator_and_distill():
    rssm = StubRSSM()
    actor = StubActor()

    def value_fn(z):
        return torch.tensor(0.0, device=z.device)

    z_saved = torch.zeros((1, 8))

    class Cfg:
        B = 2
        H = 3
        kappa = 0.1

        class cem:
            enabled = False

    branches = simulate_branches_advanced(rssm, actor, value_fn, z_saved, Cfg)
    assert len(branches) == 2
    actions, probs = branch_to_action_distribution(branches, topk_frac=0.5)
    # distribution may be None if no actions, but should be valid here
    assert actions is not None and probs is not None
    a = select_branch_action(branches)
    assert a is not None













