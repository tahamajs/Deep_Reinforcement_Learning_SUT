import torch
import copy

from ..src.models.q_ensemble import QEnsemble


def test_soft_update_from_moves_target_towards_source():
    s_dim = 4
    a_dim = 2
    src = QEnsemble(s_dim, a_dim, num_q=2)
    tgt = QEnsemble(s_dim, a_dim, num_q=2)
    # snapshot target param
    before = {k: v.clone() for k, v in tgt.state_dict().items()}
    # modify source params deterministically
    for p in src.parameters():
        p.data.add_(1.0)
    # perform soft update with tau=0.5
    tgt.soft_update_from(src, tau=0.5)
    after = {k: v for k, v in tgt.state_dict().items()}
    # check that each parameter moved from before towards src
    for k in before:
        b = before[k]
        a = after[k]
        s = src.state_dict()[k]
        # assert a is between b and s (elementwise)
        assert torch.all((a - b) * (s - b) >= -1e-8)
