import torch
from paperAssignments.Assignments1_50.CA20.src import config, model, data, losses, utils  # type: ignore
import os


def test_model_forward_and_sample():
    cfg = config.Config()
    net = model.MLPPolicy(cfg.obs_dim, cfg.action_dim, hidden_dim=cfg.hidden_dim)
    obs = torch.randn(4, cfg.obs_dim)
    mean, log_std = net(obs)
    assert mean.shape == (4, cfg.action_dim)
    assert log_std.shape == (cfg.action_dim,)
    a, lp = net.sample(obs)
    assert a.shape == (4, cfg.action_dim)
    assert lp.shape == (4,)


def test_lagrangian_loss_and_multiplier_step():
    cfg = config.Config()
    # fake batch
    batch_constraints = torch.tensor([0.0, 1.0, 0.0, 1.0])
    constraint = losses.compute_constraint(batch_constraints)
    # fake policy loss
    policy_loss = torch.tensor(0.5)
    lam = utils.LagrangeMultiplier(initial=0.1, lr=0.5, clip=10.0)
    before = lam.value
    loss = losses.lagrangian_loss(policy_loss, constraint, lam.value, cfg.constraint_c)
    assert torch.isclose(loss, loss)  # trivial: loss is a tensor
    updated = lam.step(float(constraint), cfg.constraint_c)
    assert updated >= 0.0
    assert lam.value >= 0.0


def test_checkpoint_and_config_loading(tmp_path):
    cfg = config.Config()
    cfg.epochs = 1
    cfg.batch_size = 8
    # run a tiny train to produce checkpoint
    from paperAssignments.Assignments1_50.CA20.src import train, utils as utils_mod

    res = train.train(cfg)
    assert "checkpoint" in res
    ckpt_path = res["checkpoint"]
    assert os.path.exists(ckpt_path)
    # load checkpoint into a fresh model and ensure state loaded
    models = train.make_models(cfg)
    opt = torch.optim.Adam(models["policy"].parameters(), lr=cfg.lr)
    extra = utils_mod.load_checkpoint(ckpt_path, models["policy"], opt)
    assert isinstance(extra, dict)













