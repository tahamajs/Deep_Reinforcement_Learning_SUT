"""
Minimal SAC training loop (smoke) integrating lambda-returns critic loss and
actor update with entropy. This is a development script for CA7 experiments.
"""

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.model import RecurrentCritic, StochasticActor
from src.data import SequenceReplayBuffer
from src.losses import critic_loss_lambda
from src.sac import sac_update
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = Config(device=args.device)
    cfg.alpha = 0.1
    cfg.tau = 0.005

    device = torch.device(cfg.device)
    # twin critics
    critic1 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    critic2 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target_critic1 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(
        device
    )
    target_critic2 = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(
        device
    )
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    actor = StochasticActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)

    opt_c1 = optim.Adam(critic1.parameters(), lr=cfg.lr_critic)
    opt_c2 = optim.Adam(critic2.parameters(), lr=cfg.lr_critic)
    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)

    # automatic alpha tuning variables
    target_entropy = -float(cfg.action_dim)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=1e-3)

    buf = SequenceReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.seq_len, max_size=2000)
    # populate buffer with actor behavior
    for _ in range(500):
        obs = np.random.randn(cfg.seq_len, cfg.obs_dim).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.tensor(obs[None, ...], device=device)
            acts_t, logp_t, _ = actor.sample(obs_t)
        acts = acts_t[0].cpu().numpy().astype(np.float32)
        logp = logp_t[0].cpu().numpy().astype(np.float32)
        rews = np.random.randn(cfg.seq_len).astype(np.float32) * 0.1
        dones = np.zeros(cfg.seq_len, dtype=np.float32)
        buf.add(obs, acts, rews, dones, beh_logp=logp)

    for step in range(100):
        batch = buf.sample_batch(cfg.batch_size, device=str(device))
        # critic updates
        loss_c1, returns = critic_loss_lambda(
            critic1,
            target_critic1,
            *batch,
            cfg.gamma,
            cfg.lam,
            c_rho=cfg.c_rho,
            policy=actor,
        )
        opt_c1.zero_grad()
        loss_c1.backward()
        nn.utils.clip_grad_norm_(critic1.parameters(), cfg.grad_clip)
        opt_c1.step()

        loss_c2, _ = critic_loss_lambda(
            critic2,
            target_critic2,
            *batch,
            cfg.gamma,
            cfg.lam,
            c_rho=cfg.c_rho,
            policy=actor,
        )
        opt_c2.zero_grad()
        loss_c2.backward()
        nn.utils.clip_grad_norm_(critic2.parameters(), cfg.grad_clip)
        opt_c2.step()

        # actor update (uses both critics)
        a_loss = sac_update(
            [critic1, critic2],
            [target_critic1, target_critic2],
            actor,
            [opt_c1, opt_c2],
            opt_a,
            batch,
            cfg,
        )

        # automatic alpha tuning
        with torch.no_grad():
            actions_pi, logp_pi, _ = actor.sample(batch[0])
        alpha = log_alpha.exp()
        alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
        opt_alpha.zero_grad()
        alpha_loss.backward()
        opt_alpha.step()

        if step % 10 == 0:
            print(
                f"step={step} c1={loss_c1.item():.6f} c2={loss_c2.item():.6f} a_loss={a_loss:.6f} alpha={alpha.item():.4f}"
            )


if __name__ == "__main__":
    main()

