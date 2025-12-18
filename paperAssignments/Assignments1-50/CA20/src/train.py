from __future__ import annotations
from typing import Dict, Any
import os
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .data import make_dataloader
from .model import MLPPolicy, ValueNet
from .losses import policy_loss_from_logprob, compute_constraint, lagrangian_loss
from .utils import set_seed, LagrangeMultiplier, save_checkpoint


def make_models(cfg: Config) -> Dict[str, Any]:
    policy = MLPPolicy(cfg.obs_dim, cfg.action_dim, hidden_dim=cfg.hidden_dim)
    value = ValueNet(cfg.obs_dim, hidden_dim=cfg.hidden_dim)
    return {"policy": policy, "value": value}


def train_one_epoch(
    models: Dict[str, Any],
    optimizers: Dict[str, Any],
    dataloader,
    lagrange: LagrangeMultiplier,
    cfg: Config,
) -> Dict[str, float]:
    policy = models["policy"]
    value = models["value"]
    policy_opt = optimizers["policy"]
    value_opt = optimizers["value"]

    policy.train()
    value.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_constraint = 0.0
    n = 0

    for batch in dataloader:
        obs = batch["obs"].float()
        rewards = batch["reward"].float()
        constraints = batch["constraint"].float()

        # forward
        actions, logp = policy.sample(obs)
        values = value(obs)

        # simple advantage: reward - value
        advantages = rewards - values.detach()

        p_loss = policy_loss_from_logprob(logp, advantages)
        c_val = compute_constraint(constraints)
        lag_loss = lagrangian_loss(p_loss, c_val, lagrange.value, cfg.constraint_c)

        # value loss (MSE)
        v_loss = nn.functional.mse_loss(values, rewards)

        # update policy (with lagrangian)
        policy_opt.zero_grad()
        lag_loss.backward(retain_graph=True)
        policy_opt.step()

        # update value
        value_opt.zero_grad()
        v_loss.backward()
        value_opt.step()

        # update multiplier
        lagrange.step(float(c_val), cfg.constraint_c)

        batch_size = obs.shape[0]
        total_policy_loss += float(p_loss.item()) * batch_size
        total_value_loss += float(v_loss.item()) * batch_size
        total_constraint += float(c_val) * batch_size
        n += batch_size

    return {
        "policy_loss": total_policy_loss / max(1, n),
        "value_loss": total_value_loss / max(1, n),
        "constraint": total_constraint / max(1, n),
        "lagrange": lagrange.value,
    }


def train(cfg: Config | None = None) -> Dict[str, Any]:
    if cfg is None:
        # try to load debug config if present
        try:
            cfg = Config.from_yaml(
                os.path.join(os.path.dirname(__file__), "..", "configs", "debug.yaml")
            )
        except Exception:
            cfg = Config()

    set_seed(cfg.seed)
    dataloader = make_dataloader(
        batch_size=cfg.batch_size, obs_dim=cfg.obs_dim, action_dim=cfg.action_dim
    )
    models = make_models(cfg)
    policy_opt = optim.Adam(models["policy"].parameters(), lr=cfg.lr)
    value_opt = optim.Adam(models["value"].parameters(), lr=cfg.lr)
    lagrange = LagrangeMultiplier(initial=1.0, lr=cfg.lambda_lr, clip=cfg.lambda_clip)

    metrics = {"history": []}
    for epoch in range(cfg.epochs):
        m = train_one_epoch(
            models,
            {"policy": policy_opt, "value": value_opt},
            dataloader,
            lagrange,
            cfg,
        )
        metrics["history"].append(m)

    # save a tiny checkpoint
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "ca20_checkpoint.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_checkpoint(
        out, models["policy"], policy_opt, extra={"lagrange": lagrange.state_dict()}
    )
    metrics["checkpoint"] = out
    return metrics


if __name__ == "__main__":
    cfg = None
    try:
        cfg = Config.from_yaml(
            os.path.join(os.path.dirname(__file__), "..", "configs", "debug.yaml")
        )
    except Exception:
        cfg = Config()
    print("Starting debug training with config:", cfg)
    res = train(cfg)
    print("Training finished. Summary:", res)
