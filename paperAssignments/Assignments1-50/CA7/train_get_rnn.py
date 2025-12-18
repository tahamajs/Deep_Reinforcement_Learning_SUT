"""
Minimal training loop demonstrating lambda-return critic update using the
modules in `src/`. This script is intentionally small and safe to run for
smoke-testing. It does not perform heavy training; rather it shows how to
wire the components together.
"""

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.model import RecurrentCritic, SimpleActor
from src.data import SequenceReplayBuffer
from src.losses import critic_loss_lambda
import numpy as np


def build_dummy_buffer(
    cfg: Config, actor=None, n_sequences: int = 200, device: str = "cpu"
) -> SequenceReplayBuffer:
    buf = SequenceReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.seq_len, max_size=10000)
    for _ in range(n_sequences):
        obs = np.random.randn(cfg.seq_len, cfg.obs_dim).astype(np.float32)
        if actor is None:
            acts = np.random.uniform(
                -1.0, 1.0, size=(cfg.seq_len, cfg.action_dim)
            ).astype(np.float32)
            beh_logp = None
        else:
            # sample actions and behavior logp from actor
            with torch.no_grad():
                obs_t = torch.tensor(obs[None, ...], device=device)
                acts_t, logp_t, _ = actor.sample(obs_t)
            acts = acts_t[0].cpu().numpy().astype(np.float32)
            beh_logp = logp_t[0].cpu().numpy().astype(np.float32)
        rews = np.random.randn(cfg.seq_len).astype(np.float32) * 0.1
        dones = np.zeros(cfg.seq_len, dtype=np.float32)
        buf.add(obs, acts, rews, dones, beh_logp=beh_logp)
    return buf


def train_step(critic, target_critic, opt, batch, cfg: Config, policy=None):
    obs, acts, rews, dones, beh_logp = batch
    loss, returns = critic_loss_lambda(
        critic,
        target_critic,
        obs,
        acts,
        rews,
        dones,
        beh_logp,
        cfg.gamma,
        cfg.lam,
        c_rho=cfg.c_rho,
        policy=policy,
    )
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
    opt.step()
    return loss.item()


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = Config(device=args.device)

    device = torch.device(cfg.device)
    critic = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(device)
    target_critic = RecurrentCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_size).to(
        device
    )
    target_critic.load_state_dict(critic.state_dict())
    opt = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # create actor and buffer with behavior logprobs
    actor = StochasticActor(cfg.obs_dim, cfg.action_dim, cfg.hidden_size)
    actor.to(device)
    buf = build_dummy_buffer(cfg, actor=actor, n_sequences=200, device=str(device))

    # simple loop
    for step in range(10):
        batch = buf.sample_batch(cfg.batch_size, device=str(device))
        loss = train_step(critic, target_critic, opt, batch, cfg, policy=actor)
        if step % 2 == 0:
            # soft update
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.mul_(0.995)
                tp.data.add_(0.005 * p.data)
        print(f"step={step} loss={loss:.6f}")


if __name__ == "__main__":
    main()
















