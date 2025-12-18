"""
Dreamer Agent for Planning in Latent Space

This module implements the Dreamer agent, a model-based reinforcement learning
algorithm that learns world models and plans in latent space for sample-efficient
reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .latent_actor import LatentActor
from .latent_critic import LatentCritic
from ..experiments.config import AGENT_CONFIG, DREAMER_CONFIG, GLOBAL_CONFIG
from ..models.rssm import RSSM
from collections import deque
import random


class LatentActorCritic(nn.Module):
    """Complete latent actor-critic"""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = LatentActor(latent_dim, action_dim, hidden_dim)
        self.critic = LatentCritic(latent_dim, hidden_dim)

    def act(self, latent: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action in latent space"""
        if deterministic:
            mean, _ = self.actor(latent)
            return torch.tanh(mean)
        else:
            action, _ = self.actor.sample(latent)
            return action

    def evaluate(
        self, latents: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions in latent space"""
        mean, log_std = self.actor(latents)
        std = torch.exp(log_std)

        normal = Normal(mean, std)
        log_prob = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        values = self.critic(latents)

        return log_prob, values


class DreamerAgent:
    """Complete Dreamer agent implementation"""

    def __init__(
        self, obs_dim: int, action_dim: int, global_config: Any
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = global_config.dreamer_config.agent_config.latent_dim
        self.imagination_horizon = global_config.dreamer_config.agent_config.imagination_horizon
        self.device = global_config.device
        # simple global step counter (used by planner hooks)
        self.step = 0

        # World model components
        self.rssm = RSSM(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=global_config.dreamer_config.rssm_config.latent_dim,
            hidden_dim=global_config.dreamer_config.rssm_config.hidden_dim,
            stochastic_size=global_config.dreamer_config.rssm_config.stochastic_size,
        ).to(self.device)

        # Actor-critic in latent space
        self.actor_critic = LatentActorCritic(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=global_config.dreamer_config.agent_config.hidden_dim,
        ).to(self.device)

        # Experience buffer
        self.buffer = deque(maxlen=100000)

        # Optimizers
        self.world_optimizer = optim.Adam(self.rssm.parameters(), lr=global_config.dreamer_config.rssm_config.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=global_config.dreamer_config.agent_config.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.actor_critic.critic.parameters(), lr=global_config.dreamer_config.agent_config.critic_lr
        )
        # Planner integration (optional)
        self.planner_available = False
        self.checkpoint_buffer = None
        self.last_planner_trigger = None
        try:
            # try importing planner package (CA13)
            from planner import CheckpointBuffer, simulate_branches, should_trigger, TriggerConfig  # type: ignore

            self.planner_available = True
            # planner configuration may live on global_config.planner or use defaults
            planner_cfg = getattr(global_config, "planner", {}) or {}
            self._planner_cfg = planner_cfg
            buffer_size = int(planner_cfg.get("buffer_size", 1024))
            self.checkpoint_buffer = CheckpointBuffer(capacity=buffer_size, device=self.device)
            # simple TriggerConfig instance for thresholding; user can replace with richer config
            self._planner_trigger_cfg = TriggerConfig(
                cooldown=int(planner_cfg.get("cooldown", 8)),
                trigger_td=float(planner_cfg.get("trigger_td", 0.7)),
                trigger_unc=float(planner_cfg.get("trigger_unc", 0.2)),
                trigger_ent_low=float(planner_cfg.get("trigger_entropy_low", 0.3)),
                trigger_ent_high=float(planner_cfg.get("trigger_entropy_high", 2.0)),
            )
            # keep references to functions
            self._simulate_branches = simulate_branches
            self._should_trigger = should_trigger
        except Exception:
            # planner not available in PYTHONPATH
            self.planner_available = False
            self.checkpoint_buffer = None
            self._simulate_branches = None
            self._should_trigger = None

    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select action using current world model"""
        # Encode observation
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            post_params = self.rssm.obs_encoder(obs_tensor)
            latent = self.rssm._sample_latent(*post_params.chunk(2, dim=-1))

            # Get action from actor
            action = self.actor_critic.act(latent, deterministic)

        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ):
        """Store transition in buffer"""
        self.buffer.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
            }
        )
        # advance global step counter for planner bookkeeping
        self.step += 1

    def update_world_model(self, batch_size: int) -> Dict[str, float]:
        """Update world model using data from buffer"""
        if len(self.buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(list(self.buffer), batch_size)
        obs_batch = torch.tensor(np.array([t["obs"] for t in batch]), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array([t["action"] for t in batch]), dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor([t["reward"] for t in batch], dtype=torch.float32).to(self.device)
        # next_obs_batch = torch.tensor(np.array([t["next_obs"] for t in batch]), dtype=torch.float32).to(self.device)

        self.world_optimizer.zero_grad()

        # Process sequence (simplified single-step for now)
        det_state = torch.zeros(batch_size, self.rssm.deterministic_size, device=self.device)
        prev_latent = torch.zeros(batch_size, self.rssm.stochastic_size, device=self.device)

        # Observation step
        (
            det_state,
            post_latent,
            obs_recon,
            reward_pred,
            continue_pred,
            prior_mean,
            prior_log_var,
            post_mean,
            post_log_var,
        ) = self.rssm.observe_step(obs_batch, det_state, action_batch, prev_latent)

        # Losses
        obs_loss = F.mse_loss(obs_recon, obs_batch)
        reward_loss = F.mse_loss(reward_pred.squeeze(), reward_batch)
        continue_loss = F.binary_cross_entropy(
            continue_pred.squeeze(),
            torch.tensor([not t["done"] for t in batch], dtype=torch.float32, device=self.device),
        )

        # KL divergence between posterior and prior
        kl_loss = self._kl_divergence(
            post_mean, post_log_var, prior_mean, prior_log_var
        )

        total_loss = obs_loss + reward_loss + continue_loss + kl_loss

        total_loss.backward()
        self.world_optimizer.step()

        return {
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_world_loss": total_loss.item(),
        }
        # Planner hook: compute simple TD-like signal (reward prediction error)
        # and add latent checkpoints for high-error samples. This block is intentionally
        # lightweight and robust to missing planner modules.
        try:
            if self.planner_available and self.checkpoint_buffer is not None:
                # reward_pred: [batch, 1], reward_batch: [batch]
                td_vec = torch.abs(reward_pred.squeeze() - reward_batch).detach()
                # threshold can be configured via global_config.planner.threshold; fallback 0.5
                td_thresh = float(getattr(global_config, "planner", {}).get("trigger_td", 0.7))
                # post_latent corresponds to per-sample posterior latent (batch x latent_dim)
                for i, td in enumerate(td_vec):
                    if float(td) >= td_thresh:
                        z_i = post_latent[i].detach()
                        self.checkpoint_buffer.push(z_i, score=float(td), step=self.step)
                # Optionally trigger planner immediately if condition met on mean td
                td_mean = float(td_vec.mean().item())
                if self._should_trigger is not None and self._simulate_branches is not None:
                    if self._should_trigger(td_mean, 0.0, 0.0, self._planner_trigger_cfg, self.last_planner_trigger, self.step):
                        # sample a checkpoint and run simulated branches (non-blocking, no-grad)
                        samples = self.checkpoint_buffer.sample(k=1, prioritized=True) if len(self.checkpoint_buffer) > 0 else []
                        if samples:
                            z_saved = samples[0]["z"]
                            # value function wrapper
                            def value_fn(z):
                                with torch.no_grad():
                                    v = self.actor_critic.critic(z.view(z.shape[0], -1))
                                    # return scalar for bootstrap; if batch, take mean
                                    return v.mean()

                            branches = self._simulate_branches(self.rssm, self.actor_critic.actor, value_fn, z_saved, self._planner_cfg)
                            # store metadata and update last trigger
                            self.last_planner_trigger = self.step
                            # Update actor/critic from branches (uses TD(lambda) aggregator)
                            try:
                                losses = self.update_actor_critic_from_branches(branches)
                                if losses:
                                    print(f\"[Planner] step={self.step} planner update actor_loss={losses.get('actor_loss'):.4f} critic_loss={losses.get('critic_loss'):.4f}\")
                            except Exception:
                                # fallback logging if update fails
                                if branches:
                                    top_ret = branches[0].ret
                                    print(f\"[Planner] step={self.step} triggered. top_branch_return={top_ret:.3f}\")
        except Exception:
            # swallow planner exceptions to avoid breaking world-model training
            pass

    def _kl_divergence(
        self,
        mean1: torch.Tensor,
        log_var1: torch.Tensor,
        mean2: torch.Tensor,
        log_var2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between two Gaussians"""
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)

        kl = 0.5 * (log_var2 - log_var1 + (var1 + (mean1 - mean2).pow(2)) / var2 - 1)
        return kl.sum()

    def update_actor_critic(self, batch_size: int) -> Dict[str, float]:
        """Update actor-critic using imagination"""
        if len(self.buffer) < batch_size:
            return {}

        # Sample initial states from buffer
        batch = random.sample(list(self.buffer), batch_size)
        obs_batch = torch.tensor(np.array([t["obs"] for t in batch]), dtype=torch.float32).to(self.device)

        # Encode initial observations
        with torch.no_grad():
            post_params = self.rssm.obs_encoder(obs_batch)
            init_latent = self.rssm._sample_latent(*post_params.chunk(2, dim=-1))
            init_det_state = torch.zeros(batch_size, self.rssm.deterministic_size, device=self.device)

        # Imagine trajectories
        imagined_latents = []
        imagined_rewards = []
        imagined_actions = []
        imagined_log_probs = []

        latent = init_latent
        det_state = init_det_state

        for _ in range(self.imagination_horizon):
            # Sample action
            action, log_prob = self.actor_critic.actor.sample(latent)

            # Imagine next state
            det_state, latent, reward, _ = self.rssm.imagine_step(
                det_state, action, latent
            )

            imagined_latents.append(latent)
            imagined_rewards.append(reward)
            imagined_actions.append(action)
            imagined_log_probs.append(log_prob)

        # Stack imagined trajectory
        imagined_latents = torch.stack(imagined_latents)  # [horizon, batch, latent_dim]
        imagined_rewards = torch.stack(imagined_rewards)  # [horizon, batch, 1]
        imagined_actions = torch.stack(imagined_actions)  # [horizon, batch, action_dim]
        imagined_log_probs = torch.stack(imagined_log_probs)  # [horizon, batch, 1]

        # Compute returns
        returns = self._compute_returns(imagined_rewards, gamma=AGENT_CONFIG.gamma)

        # Update critic
        self.critic_optimizer.zero_grad()
        values = self.actor_critic.critic(imagined_latents.view(-1, self.latent_dim))
        critic_loss = F.mse_loss(values, returns.view(-1, 1))
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        advantages = returns - values.detach()
        actor_loss = -(imagined_log_probs.view(-1) * advantages.view(-1)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()
        }

    def update_actor_critic_from_branches(self, branches: List[Any], topk_frac: float = 0.5, lambda_: float = 0.95) -> Dict[str, float]:
        """
        Update actor and critic from simulated branches (produced by simulate_branches).
        branches: list of Branch objects with .traj = list of (z, a, r, gamma)
        This implements TD(lambda) targets on imagined rollouts and performs one
        gradient step on critic and actor using the aggregated top-k branches.
        """
        if not branches:
            return {}
        k = max(1, int(len(branches) * topk_frac))
        selected = branches[:k]

        # collect flattened lists across branches
        z_list = []
        a_list = []
        r_list = []
        gamma_list = []
        lengths = []
        for br in selected:
            traj = br.traj
            lengths.append(len(traj))
            for (z, a, r, gamma) in traj:
                # ensure z is 2D (batch style)
                if isinstance(z, torch.Tensor) and z.dim() == 1:
                    z = z.unsqueeze(0)
                z_list.append(z)
                # actions may not be tensors; convert
                if isinstance(a, torch.Tensor):
                    a_list.append(a.view(-1))
                else:
                    a_list.append(torch.tensor(a, dtype=torch.float32))
                r_list.append(float(r))
                gamma_list.append(float(gamma))

        if len(z_list) == 0:
            return {}

        # Stack tensors
        z_batch = torch.cat([z for z in z_list], dim=0).to(self.device)  # [N, latent_dim]
        a_batch = torch.stack([a for a in a_list]).to(self.device)  # [N, action_dim]
        r_tensor = torch.tensor(r_list, dtype=torch.float32, device=self.device)
        gamma_tensor = torch.tensor(gamma_list, dtype=torch.float32, device=self.device)

        # Compute values for each z (bootstrap)
        with torch.no_grad():
            values = self.actor_critic.critic(z_batch).view(-1)  # [N]

        # Compute TD(lambda) targets per timestep following branch order
        # We need to reconstruct per-branch indexing to compute returns with bootstraps.
        targets = []
        idx = 0
        for L in lengths:
            # for branch with length L, compute G_t backwards
            G_next = 0.0
            # bootstrap with value at final latent
            if L > 0:
                last_idx = idx + L - 1
                G_next = float(values[last_idx].item())
            Gs = [0.0] * L
            for t in range(L - 1, -1, -1):
                r_t = float(r_tensor[idx + t].item())
                g_t = float(gamma_tensor[idx + t].item())
                G_t = r_t + g_t * ((1.0 - lambda_) * (float(values[idx + t + 1].item()) if (t < L - 1) else 0.0) + lambda_ * G_next) if L > 0 else r_t
                Gs[t] = G_t
                G_next = G_t
            targets.extend(Gs)
            idx += L

        targets = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)  # [N,1]

        # Critic update
        try:
            self.critic_optimizer.zero_grad()
            pred_vals = self.actor_critic.critic(z_batch)
            critic_loss = F.mse_loss(pred_vals, targets)
            critic_loss.backward()
            self.critic_optimizer.step()
        except Exception:
            critic_loss = torch.tensor(0.0)

        # Actor update (policy gradient using advantage = G - V)
        try:
            self.actor_optimizer.zero_grad()
            # compute log probs for actions under current policy
            logp, _ = self.actor_critic.evaluate(z_batch, a_batch)
            with torch.no_grad():
                new_values = self.actor_critic.critic(z_batch).detach()
                advantages = targets - new_values
            actor_loss = -(logp.view(-1, 1) * advantages).mean()
            actor_loss.backward()
            self.actor_optimizer.step()
        except Exception:
            actor_loss = torch.tensor(0.0)

        return {"actor_loss": float(actor_loss) if isinstance(actor_loss, torch.Tensor) else actor_loss, "critic_loss": float(critic_loss) if isinstance(critic_loss, torch.Tensor) else critic_loss}

    def _compute_returns(
        self, rewards: torch.Tensor, gamma: float = 0.99
    ) -> torch.Tensor:
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.stack(returns)
