import torch
from paperAssignments.Assignments1_50.CA15.src import Config, MLPPolicy, ValueNetwork, set_seed


def test_import_and_forward():
    cfg = Config()
    set_seed(cfg.seed)
    policy = MLPPolicy(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
    value = ValueNetwork(cfg.input_dim, cfg.hidden_dim)
    x = torch.randn(2, cfg.input_dim)
    logits = policy(x)
    assert logits.shape == (2, cfg.output_dim)
    v = value(x)
    assert v.shape == (2,)

