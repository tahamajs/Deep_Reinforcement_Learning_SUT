import torch
from planner.simgolf_latent import simulate_branches


class StubRSSM:
    def step(self, z, a):
        z_next = z + 0.1
        r = 1.0
        gamma = 1.0
        return z_next, r, gamma


class StubActor:
    def sample(self, z):
        return torch.ones((z.shape[0], 1), device=z.device)


def test_simulate_branches_returns():
    rssm = StubRSSM()
    actor = StubActor()

    def v(z):
        return torch.tensor(0.0, device=z.device)

    z_saved = torch.zeros((1, 8))

    class Cfg:
        B = 3
        H = 4

    branches = simulate_branches(rssm, actor, v, z_saved, Cfg)
    assert len(branches) == 3
    assert all(hasattr(b, "ret") for b in branches)
    rets = [b.ret for b in branches]
    assert sorted(rets, reverse=True) == rets















