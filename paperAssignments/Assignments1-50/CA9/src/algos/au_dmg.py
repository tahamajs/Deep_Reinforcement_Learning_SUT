from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..models.q_ensemble import QEnsemble
from ..models.behavior_cvae import CVAE
from ..models.value_net import ValueNet, expectile_loss
from ..models.policy import GaussianPolicy


class AUDMG:
    """A compact, runnable implementation of AU-DMG training update logic."""

    def __init__(self, s_dim: int, a_dim: int, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # models
        self.q = QEnsemble(s_dim, a_dim, num_q=cfg.ensemble_size).to(self.device)
        self.q_targ = QEnsemble(s_dim, a_dim, num_q=cfg.ensemble_size).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.cvae = CVAE(s_dim, a_dim, cfg.latent_dim).to(self.device)
        self.value = ValueNet(s_dim).to(self.device)
        self.policy = GaussianPolicy(s_dim, a_dim).to(self.device)

        # optimizers
        self.opt_q = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.opt_value = optim.Adam(self.value.parameters(), lr=cfg.lr)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_cvae = optim.Adam(self.cvae.parameters(), lr=cfg.lr)

    def _compute_targets(self, s_next: torch.Tensor, r: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Vectorized target computation following README pseudocode."""
        B = s_next.shape[0]
        M = self.cfg.candidate_M
        a_cand = self.cvae.sample(s_next, M)  # [B, M, a_dim]
        noise = torch.randn_like(a_cand) * self.cfg.candidate_sigma
        a_cand = torch.clamp(a_cand + noise, -self.cfg.eps, self.cfg.eps)
        # evaluate target ensemble on candidates
        N = self.cfg.ensemble_size
        q_stack = torch.stack([qk.forward(s_next.repeat_interleave(M, dim=0), a_cand.reshape(B * M, -1))[0]
                               for qk in self.q_targ.qs], dim=0)  # [N, B*M, 1]
        q_stack = q_stack.view(N, B, M, 1)
        q_mean = q_stack.mean(0)  # [B, M, 1]
        q_std = q_stack.std(0, unbiased=False)  # [B, M, 1]
        idx = q_mean.squeeze(-1).argmax(dim=1)  # [B]
        a_mild = a_cand[torch.arange(B), idx]  # [B, a_dim]
        std_mild = q_std.squeeze(-1)[torch.arange(B), idx]  # [B]
        # compute lambda (sigmoid gate)
        lam = torch.sigmoid((self.cfg.kappa / (std_mild + 1e-6)) - self.cfg.beta).unsqueeze(-1)  # [B,1]
        # y_mild from target ensemble mean evaluated at a_mild
        y_mild_stack = torch.stack([qk.forward(s_next, a_mild)[0] for qk in self.q_targ.qs], dim=0)
        y_mild = y_mild_stack.mean(0)  # [B,1]
        # y_in via value network
        y_in = self.value(s_next).detach()  # [B,1]
        y = r.unsqueeze(-1) + self.cfg.gamma * (1.0 - done.unsqueeze(-1)) * (lam * y_mild + (1.0 - lam) * y_in)
        return y, lam, std_mild

    def update(self, batch: Dict[str, torch.Tensor]):
        """One training update given a batch dict with keys: s,a,r,s_next,done"""
        s = batch["s"].to(self.device)
        a = batch["a"].to(self.device)
        r = batch["r"].to(self.device)
        s_next = batch["s_next"].to(self.device)
        done = batch["done"].to(self.device)

        # 1) Update value network (IQL expectile)
        with torch.no_grad():
            q_targ_min = torch.min(torch.stack([qk.forward(s, a)[0] for qk in self.q_targ.qs], dim=0), dim=0).values
        v_loss = expectile_loss(q_targ_min - self.value(s), tau=0.7)
        self.opt_value.zero_grad()
        v_loss.backward()
        self.opt_value.step()

        # 2) Construct targets
        y, lam, std_mild = self._compute_targets(s_next, r, done)

        # 3) Update critics
        q_vals = self.q.all_q(s, a).squeeze(-1)  # [N, B]
        Y = y.detach().squeeze(-1)  # [B]
        critic_loss = ((q_vals - Y.unsqueeze(0)) ** 2).mean()
        self.opt_q.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt_q.step()

        # 4) Update policy (AWR-style)
        with torch.no_grad():
            q_on_a = self.q.forward(s, a)[0].squeeze(-1)
            v_s = self.value(s).squeeze(-1)
            weights = torch.exp(torch.clamp(1.0 * (q_on_a - v_s), min=-50.0, max=50.0)).unsqueeze(-1)
        logp = self.policy.log_prob(s, a).unsqueeze(-1)
        policy_loss = -(weights * logp).mean()
        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        # 5) Soft update targets
        self.q_targ.soft_update_from(self.q, self.cfg.tau)

        return {
            "v_loss": v_loss.item(),
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "lam_mean": lam.mean().item(),
            "std_mild_mean": std_mild.mean().item(),
        }

