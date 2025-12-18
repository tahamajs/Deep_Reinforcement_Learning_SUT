import os
import tempfile
import torch
import numpy as np

from paperAssignments.Assignments1_50.CA9.src.config import default_config
from paperAssignments.Assignments1_50.CA9.src.algos.au_dmg import AUDMG


def test_checkpoint_roundtrip():
    cfg = default_config()
    s_dim = 6
    a_dim = 2
    agent = AUDMG(s_dim, a_dim, cfg)
    # make a determinisitic change
    for p in agent.policy.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)
    fd, path = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    try:
        agent.save_checkpoint(path)
        # mutate parameters
        before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
        for p in agent.policy.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        # load
        agent.load_checkpoint(path, map_location="cpu")
        after = agent.policy.state_dict()
        # check that policy params restored to saved (close)
        for k in before:
            assert torch.allclose(before[k], after[k], atol=1e-6, rtol=1e-4)
    finally:
        os.remove(path)










