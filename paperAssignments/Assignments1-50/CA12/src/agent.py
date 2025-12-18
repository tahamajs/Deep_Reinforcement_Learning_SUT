from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim

from .models import GaussianActor, VectorizedCritic
from .retrieval_buffer import RetrievalBuffer
from .config import Config


class RAUOBACAgent:
    """Retrieval-Augmented Uncertainty-aware OBAC agent (minimal, functional implementation)."""

    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.cfg = cfg
        device = torch.device(cfg.device)
        self.device = device

        # networks
        self.actor = GaussianActor(state_dim, action_dim).to(device)
        self.critic = VectorizedCritic(
            state_dim, action_dim, ensemble_size=cfg.critic_ensemble_size
        ).to(device)
        self.critic_target = VectorizedCritic(
            state_dim, action_dim, ensemble_size=cfg.critic_ensemble_size
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # offline actor (parametric baseline) - optional; we keep a clone for completeness
        self.offline_actor = GaussianActor(state_dim, action_dim).to(device)

        # optimizers
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.offline_opt = optim.Adam(
            self.offline_actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # buffer
        # pass a torch.device to the buffer (it normalizes internally)
        self.retrieval_buffer = RetrievalBuffer(
            cfg.buffer_size, state_dim, action_dim, device=device
        )

    def update_critic(
        self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor
    ) -> float:
        """Update ensemble critics using MSE to return-to-go as a simple target."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        qs = self.critic(states, actions)  # (E, B, 1)
        # target: use mean over ensemble target networks
        with torch.no_grad():
            target_qs = self.critic_target(states, actions)  # (E, B, 1)
            target = returns.unsqueeze(0)  # (1, B, 1)
            target = target.expand_as(target_qs)

        loss = F.mse_loss(qs, target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        # soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.cfg.tau)
            tp.data.add_(self.cfg.tau * p.data)
        return float(loss.item())

    def update_offline_actor(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: Optional[torch.Tensor] = None,
    ) -> float:
        """AWR-style update: weighted behavior cloning towards high-advantage actions."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        if advantages is None:
            weights = torch.ones((states.shape[0], 1), device=self.device)
        else:
            weights = (
                torch.exp(torch.clamp(advantages, max=50.0))
                .unsqueeze(1)
                .to(self.device)
            )

        mean, log_std = self.offline_actor(states)
        # Gaussian MLE loss (squared error on means as proxy)
        loss = (weights * (mean - actions).pow(2)).mean()
        self.offline_opt.zero_grad()
        loss.backward()
        self.offline_opt.step()
        return float(loss.item())

    def compute_uncertainty_penalized_value(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute ensemble mean minus beta * std across critics for a single (state,action)."""
        qs = self.critic(state.unsqueeze(0), action.unsqueeze(0)).squeeze(-1)  # (E, 1)
        mean = qs.mean(dim=0)  # (1,)
        std = qs.std(dim=0)  # (1,)
        val = mean - self.cfg.beta_uq * std
        return val.squeeze(0)

    def update_online_actor(self, states: torch.Tensor) -> float:
        """Update online actor with SAC-like loss plus retrieval-based boosting when mask active.

        This routine:
        - samples actions from actor
        - computes SAC surrogate (using min ensemble Q)
        - retrieves best-k actions for each state, computes LCB value, compares to online value and applies boosting loss if better
        """
        states = states.to(self.device)
        batch_size = states.shape[0]

        # sample actions and log probs
        with torch.no_grad():
            actions_sampled, logp = self.actor.sample(states)

        # compute Q estimates for sampled actions
        qs_sampled = self.critic(states, actions_sampled).squeeze(-1)  # (E, B)
        q_min = qs_sampled.min(dim=0)[0]  # (B,)
        sac_loss = (logp.squeeze(-1) - q_min).mean()

        # retrieval and boosting
        boost_losses = []
        masks = []
        for i in range(batch_size):
            s = states[i].detach().cpu()
            retrieved_actions = self.retrieval_buffer.retrieve_best_k(
                s, k=self.cfg.retrieval_k, nn=self.cfg.retrieval_nn
            )
            if retrieved_actions.shape[0] == 0:
                masks.append(0.0)
                boost_losses.append(torch.tensor(0.0, device=self.device))
                continue
            # evaluate retrieved actions with ensemble to get LCB
            retrieved_actions = retrieved_actions.to(self.device)
            # expand state to match actions
            s_exp = states[i].unsqueeze(0).expand(retrieved_actions.shape[0], -1)
            qs = self.critic(s_exp, retrieved_actions).squeeze(-1)  # (E, k)
            mu_q = qs.mean(dim=0)  # (k,)
            std_q = qs.std(dim=0)  # (k,)
            lcb = mu_q - self.cfg.beta_uq * std_q
            v_target, idx = torch.max(lcb, dim=0)

            # online value estimate: use min ensemble Q for actor action
            with torch.no_grad():
                a_online, _ = self.actor.sample(states[i].unsqueeze(0))
                q_online = self.critic(states[i].unsqueeze(0), a_online).squeeze(
                    -1
                )  # (E,1)
                v_online = (q_online.min(dim=0)[0]).squeeze(0)

            mask = 1.0 if (v_target.item() > v_online.item()) else 0.0
            masks.append(float(mask))
            if mask > 0.0:
                a_target = retrieved_actions[idx].to(self.device)
                mean, _ = self.actor(states[i].unsqueeze(0))
                # MSE to nearest retrieved action (minimum over k already picked)
                boost_losses.append(((mean - a_target.unsqueeze(0)).pow(2).mean()))
            else:
                boost_losses.append(torch.tensor(0.0, device=self.device))

        boost_loss = (
            torch.stack(boost_losses).mean()
            if len(boost_losses) > 0
            else torch.tensor(0.0, device=self.device)
        )
        masks_mean = sum(masks) / max(len(masks), 1)
        total_loss = sac_loss + self.cfg.lambda_blend * boost_loss

        self.actor_opt.zero_grad()
        total_loss.backward()
        self.actor_opt.step()
        return float(total_loss.item())
