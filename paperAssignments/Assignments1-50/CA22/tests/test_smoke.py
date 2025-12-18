import pytest
import torch
from src.config import ExperimentConfig
from src.model import PolicyNet, ValueNet, sample_action
from src.losses import policy_gradient_loss, value_loss, LagrangianLoss
from src.data import SyntheticDataset
from src.utils import set_seed


def test_config_load():
    cfg = ExperimentConfig()
    assert cfg.obs_dim > 0


def test_model_forward():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    policy = PolicyNet(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)
    value = ValueNet(cfg.obs_dim, cfg.hidden_size)
    x = torch.randn(4, cfg.obs_dim)
    logits = policy(x)
    assert logits.shape == (4, cfg.action_dim)
    v = value(x)
    assert v.shape == (4,)
    a, lp = sample_action(logits)
    assert a.shape == (4,) and lp.shape == (4,)


def test_losses_and_dataset():
    cfg = ExperimentConfig()
    ds = SyntheticDataset(
        num_episodes=10, obs_dim=cfg.obs_dim, horizon=5, seed=cfg.seed
    )
    states, actions, rewards, constraints = ds.sample_batch(batch_size=2)
    assert states.shape[1] == cfg.obs_dim
    # make small torch tensors
    import torch

    logp = torch.randn(states.shape[0])
    adv = torch.randn_like(logp)
    pg = policy_gradient_loss(logp, adv)
    assert isinstance(pg.item(), float)
    v_pred = torch.randn(states.shape[0])
    v_targ = torch.randn(states.shape[0])
    _ = value_loss(v_pred, v_targ)
    lag = LagrangianLoss(mu=0.0, constraint_threshold=cfg.constraint_threshold)
    combined = lag(pg, torch.from_numpy(constraints))
    assert combined.shape == ()








