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
    critic = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target_critic = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target_critic.load_state_dict(critic.state_dict())
    actor = StochasticActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)

    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)
    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)

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

    for step in range(50):
        batch = buf.sample_batch(cfg.batch_size, device=str(device))
        # critic update
        loss_c, returns = critic_loss_lambda(critic, target_critic, *batch, cfg.gamma, cfg.lam, c_rho=cfg.c_rho, policy=actor)
        opt_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
        opt_c.step()

        # actor update
        a_loss = sac_update(critic, target_critic, actor, opt_c, opt_a, batch, cfg)

        if step % 5 == 0:
            print(f"step={step} critic_loss={loss_c.item():.6f} actor_loss={a_loss:.6f}")


if __name__ == "__main__":
    main()

