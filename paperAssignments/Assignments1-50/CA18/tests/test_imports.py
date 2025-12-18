import sys
from pathlib import Path

# Ensure package path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch


def test_imports_and_forward():
    import config as cfg  # type: ignore
    import model as m  # type: ignore
    import data as d  # type: ignore
    import losses as losses  # type: ignore
    import utils as utils  # type: ignore

    c = cfg.Config()
    utils.set_seed(c.seed)
    obs_dim = 4
    action_dim = 2
    net = m.ActorCritic(obs_dim, action_dim, hidden_sizes=c.hidden_sizes)
    batch = torch.randn(8, obs_dim)
    actions, logp = net.act(batch)
    assert actions.shape[0] == batch.shape[0]
    assert logp.shape[0] == batch.shape[0]
    v = net.get_value(batch)
    assert v.shape == (batch.shape[0],)


