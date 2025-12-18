"""
Full SAC-style training loop (development/demo) integrating:
- Twin critics with lambda-return critic loss
- Actor updates with automatic alpha tuning
- Checkpointing and CSV logging

This is intended as a reproducible baseline for CA7 experiments.
"""

import argparse
import time
import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.model import RecurrentCritic, StochasticActor
from src.data import SequenceReplayBuffer
from src.losses import critic_loss_lambda
from src.sac import sac_update
from src.utils import save_checkpoint, CSVLogger


def build_buffer(cfg: Config, actor: StochasticActor, device: str, n=1000):
    buf = SequenceReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.seq_len, max_size=5000)
    for _ in range(n):
        obs = torch.randn(cfg.seq_len, cfg.obs_dim).cpu().numpy().astype("float32")
        with torch.no_grad():
            obs_t = torch.tensor(obs[None], device=device)
            acts_t, logp_t, _ = actor.sample(obs_t)
        acts = acts_t[0].cpu().numpy().astype("float32")
        logp = logp_t[0].cpu().numpy().astype("float32")
        rews = (0.01 * torch.randn(cfg.seq_len)).cpu().numpy().astype("float32")
        dones = (torch.zeros(cfg.seq_len)).cpu().numpy().astype("float32")
        buf.add(obs, acts, rews, dones, beh_logp=logp)
    return buf


def train(cfg: Config, out_dir: str):
    device = torch.device(cfg.device)
    critic1 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    critic2 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target1 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target2 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())
    actor = StochasticActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)

    opt_c1 = optim.Adam(critic1.parameters(), lr=cfg.lr_critic)
    opt_c2 = optim.Adam(critic2.parameters(), lr=cfg.lr_critic)
    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)

    # alpha params
    target_entropy = -float(cfg.action_dim)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=1e-3)

    buf = build_buffer(cfg, actor, device, n=1000)

    logger = CSVLogger(
        os.path.join(out_dir, "train_log.csv"),
        header={
            "step": "step",
            "critic1_loss": "critic1_loss",
            "critic2_loss": "critic2_loss",
            "actor_loss": "actor_loss",
            "alpha": "alpha",
        },
    )

    step = 0
    max_steps = 500
    for epoch in range(0, max_steps):
        batch = buf.sample_batch(cfg.batch_size, device=str(device))
        loss_c1, returns = critic_loss_lambda(
            critic1, target1, *batch, cfg.gamma, cfg.lam, c_rho=cfg.c_rho, policy=actor
        )
        opt_c1.zero_grad()
        loss_c1.backward()
        nn.utils.clip_grad_norm_(critic1.parameters(), cfg.grad_clip)
        opt_c1.step()

        loss_c2, _ = critic_loss_lambda(
            critic2, target2, *batch, cfg.gamma, cfg.lam, c_rho=cfg.c_rho, policy=actor
        )
        opt_c2.zero_grad()
        loss_c2.backward()
        nn.utils.clip_grad_norm_(critic2.parameters(), cfg.grad_clip)
        opt_c2.step()

        a_loss = sac_update(
            [critic1, critic2],
            [target1, target2],
            actor,
            [opt_c1, opt_c2],
            opt_a,
            batch,
            cfg,
        )

        # alpha update
        with torch.no_grad():
            _, logp_pi, _ = actor.sample(batch[0])
        alpha = log_alpha.exp()
        alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
        opt_alpha.zero_grad()
        alpha_loss.backward()
        opt_alpha.step()

        if epoch % 10 == 0:
            print(
                f"epoch={epoch} c1={loss_c1.item():.6f} c2={loss_c2.item():.6f} a={a_loss:.6f} alpha={alpha.item():.4f}"
            )

        logger.log(
            {
                "step": epoch,
                "critic1_loss": float(loss_c1.item()),
                "critic2_loss": float(loss_c2.item()),
                "actor_loss": float(a_loss),
                "alpha": float(alpha.item()),
            }
        )

        # checkpoint periodically
        if epoch % 100 == 0:
            state = {
                "epoch": epoch,
                "critic1": critic1.state_dict(),
                "critic2": critic2.state_dict(),
                "actor": actor.state_dict(),
                "opt_c1": opt_c1.state_dict(),
                "opt_c2": opt_c2.state_dict(),
                "opt_a": opt_a.state_dict(),
                "log_alpha": log_alpha.detach().cpu().numpy(),
            }
            save_checkpoint(os.path.join(out_dir, f"checkpoint_{epoch}.pt"), state)

    # final save
    save_checkpoint(
        os.path.join(out_dir, "final_checkpoint.pt"),
        {
            "critic1": critic1.state_dict(),
            "critic2": critic2.state_dict(),
            "actor": actor.state_dict(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/ca7")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = Config(device=args.device)
    train(cfg, args.out)


if __name__ == "__main__":
    main()








