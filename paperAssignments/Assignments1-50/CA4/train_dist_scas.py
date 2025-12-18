import argparse
from pathlib import Path
import torch
import torch.optim as optim
from src.config import Config
from src.model import QuantileMLP, TanhGaussianPolicy, SCASReg
from src.losses import quantile_huber_loss, cvar_tail
from src.data import ReplayBuffer
from src.utils import set_seed, soft_update, iqr_from_tensor
import numpy as np


def build_models(s_dim: int, a_dim: int, cfg: Config, device: str):
    critics = [
        QuantileMLP(s_dim, a_dim, n_q=cfg.n_quantiles).to(device)
        for _ in range(cfg.n_critics)
    ]
    target_critics = [
        QuantileMLP(s_dim, a_dim, n_q=cfg.n_quantiles).to(device)
        for _ in range(cfg.n_critics)
    ]
    actor = TanhGaussianPolicy(s_dim, a_dim).to(device)
    scas = SCASReg(s_dim, a_dim).to(device)
    return critics, target_critics, actor, scas


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    # Placeholder dims for import-safety; user should replace with env-derived values
    s_dim = 17
    a_dim = 6
    critics, target_critics, actor, scas = build_models(s_dim, a_dim, cfg, device)

    # optimizers
    critic_opts = [optim.Adam(c.parameters(), lr=cfg.lr_critic) for c in critics]
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    scas_opt = optim.Adam(scas.parameters(), lr=cfg.lr_critic)

    # simple random replay for skeleton
    buf = ReplayBuffer(
        capacity=100_000, obs_dim=s_dim, act_dim=a_dim, device=str(device)
    )
    # fill with random data to keep import-safe
    buf.add_batch(
        {
            "obs": np.random.randn(1024, s_dim).astype(np.float32),
            "next_obs": np.random.randn(1024, s_dim).astype(np.float32),
            "actions": np.random.randn(1024, a_dim).astype(np.float32),
            "rewards": np.random.randn(1024).astype(np.float32),
        }
    )

    taus = (
        torch.arange(cfg.n_quantiles, device=device).float() + 0.5
    ) / cfg.n_quantiles

    for step in range(1, 3):  # minimal loop for skeleton (no heavy work)
        batch = buf.sample(cfg.batch_size if buf.size >= cfg.batch_size else 256)
        # Critic update (sketch)
        with torch.no_grad():
            a_next, _, _ = actor(batch["next_obs"])
            target_qs = [tc(batch["next_obs"], a_next) for tc in target_critics]
            pooled = torch.sort(torch.cat(target_qs, dim=1), dim=1)[0]
            y = batch["rewards"] + cfg.gamma * (1 - batch["dones"]) * pooled
        loss_c = 0.0
        for opt, c in zip(critic_opts, critics):
            opt.zero_grad()
            pred = c(batch["obs"], batch["actions"])
            loss = quantile_huber_loss(pred, y, taus, kappa=cfg.kappa)
            loss.backward()
            opt.step()
            loss_c += loss.item()

        # Actor update (sketch)
        actor_opt.zero_grad()
        a, logp, _ = actor(batch["obs"])
        qs = [c(batch["obs"], a) for c in critics]
        pooled_q = torch.sort(torch.cat(qs, dim=1), dim=1)[0]
        q_risk = cvar_tail(pooled_q, alpha=cfg.alpha_cvar)
        lam = cfg.lambda_base
        if cfg.use_adaptive_lambda:
            iqr = iqr_from_tensor(pooled_q)
            lam = cfg.lambda_base * (1 + torch.sigmoid(iqr).unsqueeze(1))
        scas_loss = scas.loss(batch["obs"], a, batch["next_obs"])
        actor_loss = (-q_risk + lam * scas_loss + cfg.entropy_beta * (-logp)).mean()
        actor_loss.backward()
        actor_opt.step()

        # Soft updates
        for tc, c in zip(target_critics, critics):
            soft_update(tc.parameters(), c.parameters(), cfg.target_tau)

    print("Training skeleton executed (no heavy training in this example).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    cfg = Config(device=args.device)
    train(cfg)










