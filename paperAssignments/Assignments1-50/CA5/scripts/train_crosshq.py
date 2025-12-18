"""Training skeleton for CrossHQ.

This script provides a ready-to-run training loop skeleton that integrates the modules
implemented in `src/crosshq`. It is intentionally non-destructive and import-safe.
"""

import argparse
import time
import torch
import torch.optim as optim
from src.crosshq.model import CrossQCritic, GaussianPolicy
from src.crosshq.losses import CrossHQLoss
from src.crosshq.relabel import off_policy_correction
from src.config import default_config


class DummyReplay:
    """Minimal replay-like object returning batched tensors for the skeleton.

    This is a placeholder so users can integrate with their preferred replay buffer.
    """

    def __init__(self, batch_size, s_dim, a_dim, c):
        self.batch = batch_size
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.c = c

    def sample_manager_batch(self):
        B = self.batch
        # Provide minimal required fields for relabeling + manager TD
        return {
            "obs": torch.randn(B, self.s_dim),
            "action_seq": torch.randn(B, self.c, self.a_dim),
            "goal": torch.randn(B, self.a_dim),
            "next_obs": torch.randn(B, self.s_dim),
        }

    def sample_worker_batch(self):
        B = self.batch
        return {
            "obs": torch.randn(B, self.s_dim),
            "action": torch.randn(B, self.a_dim),
            "reward": torch.randn(B, 1),
            "next_obs": torch.randn(B, self.s_dim),
            "mask": torch.ones(B, 1),
        }


def train_loop(cfg):
    device = torch.device(cfg.device)
    s_dim = 32
    a_dim = 6
    # Build networks
    q_lo = CrossQCritic(
        s_dim + cfg.hidden * 0,
        a_dim,
        hidden=cfg.hidden,
        depth=3,
        bn_momentum=cfg.bn_momentum,
    ).to(device)
    q_hi = CrossQCritic(
        s_dim, a_dim, hidden=cfg.hidden, depth=3, bn_momentum=cfg.bn_momentum
    ).to(device)
    pi_lo = GaussianPolicy(s_dim + a_dim, a_dim, hidden=512).to(device)
    pi_hi = GaussianPolicy(s_dim, a_dim, hidden=512).to(device)

    opt_q_lo = optim.Adam(list(q_lo.parameters()), lr=cfg.lr_critic)
    opt_q_hi = optim.Adam(list(q_hi.parameters()), lr=cfg.lr_critic)
    opt_pi_lo = optim.Adam(list(pi_lo.parameters()), lr=cfg.lr_actor)
    opt_pi_hi = optim.Adam(list(pi_hi.parameters()), lr=cfg.lr_actor)

    replay = DummyReplay(cfg.batch, s_dim, a_dim, cfg.c)

    loss_worker = CrossHQLoss(
        q_lo, pi_lo, gamma=cfg.gamma, alpha=cfg.entropy_beta, device=device
    )
    loss_manager = CrossHQLoss(
        q_hi, pi_hi, gamma=cfg.gamma**cfg.c, alpha=0.0, device=device
    )

    steps = 10  # skeleton small number; user will configure
    for step in range(steps):
        t0 = time.time()
        # Worker update
        wb = replay.sample_worker_batch()
        obs = wb["obs"].to(device)
        action = wb["action"].to(device)
        reward = wb["reward"].to(device)
        next_obs = wb["next_obs"].to(device)
        mask = wb["mask"].to(device)

        opt_q_lo.zero_grad()
        loss_q_lo = loss_worker(obs, action, reward, next_obs, mask)
        loss_q_lo.backward()
        torch.nn.utils.clip_grad_norm_(q_lo.parameters(), 10.0)
        opt_q_lo.step()

        # Worker policy update (simple deterministic ascent)
        opt_pi_lo.zero_grad()
        # actor objective: maximize Q (approximated)
        a_pi, _ = pi_lo.rsample_and_logprob(torch.cat([obs, action], dim=-1))
        # negative Q for gradient ascent
        q_val = q_lo.q1_forward(torch.cat([obs, a_pi], dim=-1))
        loss_pi_lo = -q_val.mean()
        loss_pi_lo.backward()
        opt_pi_lo.step()

        # Manager update
        mb = replay.sample_manager_batch()
        mb_device = {k: v.to(device) for k, v in mb.items()}
        g_tilde = off_policy_correction(mb_device, pi_lo)

        # construct manager tensors
        s0 = mb_device["obs"]
        g = g_tilde
        s_next = mb_device["next_obs"]
        # fake manager rewards for the skeleton
        R_hi = torch.randn(cfg.batch, 1, device=device)
        mask_hi = torch.ones(cfg.batch, 1, device=device)

        opt_q_hi.zero_grad()
        loss_q_hi = loss_manager(s0, g, R_hi, s_next, mask_hi)
        loss_q_hi.backward()
        opt_q_hi.step()

        # Manager policy update
        opt_pi_hi.zero_grad()
        g_pi, _ = pi_hi.rsample_and_logprob(s0)
        q_hi_val = q_hi.q1_forward(torch.cat([s0, g_pi], dim=-1))
        loss_pi_hi = -q_hi_val.mean()
        loss_pi_hi.backward()
        opt_pi_hi.step()

        if step % 1 == 0:
            print(
                f"step={step} loss_q_lo={loss_q_lo.item():.4f} loss_q_hi={loss_q_hi.item():.4f} time={time.time()-t0:.3f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = default_config()
    cfg.device = args.device
    train_loop(cfg)


if __name__ == "__main__":
    main()












