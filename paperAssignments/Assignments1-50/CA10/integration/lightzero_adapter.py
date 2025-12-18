"""LightZero / ma_muzero adapter for MA-EZV2.

Provides minimal adapter classes to plug MAEZV2Network and MCTS into a LightZero-style training loop.
This is a lightweight shim (no external LightZero dependency).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from paperAssignments.Assignments1_50.CA10.policy.efficientzero_v2_ma import EfficientZeroV2Policy
from paperAssignments.Assignments1_50.CA10.mcts.search import MCTS
from paperAssignments.Assignments1_50.CA10.integration.lightzero_base import BaseMuZeroPolicy


class LightZeroAdapter(BaseMuZeroPolicy):
    """Adapter that exposes methods similar to LightZero's policy interface.

    Inherits from BaseMuZeroPolicy so it can be used as a drop-in in local LightZero-like loops.
    """

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        mcfg = cfg.get("model", {})
        self.policy = EfficientZeroV2Policy(
            obs_dim=mcfg.get("obs_dim", 24),
            latent_dim=mcfg.get("latent_dim", 128),
            joint_action_dim=mcfg.get("joint_action_dim", 16),
            per_agent_action_dims=mcfg.get("per_agent_action_dims"),
            device=device,
        )
        self.mcts = MCTS(
            self.policy.net,
            c_puct=cfg.get("search", {}).get("c_puct", 1.5),
            dirichlet_alpha=cfg.get("search", {}).get("dir_alpha", None),
            dirichlet_frac=cfg.get("search", {}).get("dir_frac", 0.0),
            factored_search=cfg.get("search", {}).get("factored_search", False),
        )

    def infer(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits_joint, logits_agents, value, prefix = self.policy.initial_infer(obs)
        return {
            "logits": logits_joint,
            "logits_agents": logits_agents,
            "value": value,
            "prefix": prefix,
        }

    def search(
        self, obs: torch.Tensor, sims: int = 100, topk: int = 8
    ) -> Dict[str, torch.Tensor]:
        h0 = self.policy.net.initial_latent(obs)
        out = self.mcts.run(h0, num_simulations=sims, topk=topk)
        if isinstance(out, tuple) and len(out) == 3:
            visits, policy, joint = out
            return {"visits": visits, "policy": policy, "joint_visits": joint}
        elif isinstance(out, tuple) and len(out) == 2:
            visits, policy = out
            return {"visits": visits, "policy": policy}
        else:
            # fallback: assume dict-like
            return {"visits": out.get("visits"), "policy": out.get("policy")}

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
        optimizer: torch.optim.Optimizer,
    ):
        obs = batch["obs"]
        actions = batch["actions"]
        pi_t = batch["pi_target"]
        v_t = batch["v_target"]
        r_t = batch["r_target"]
        z_t = batch["z_target"]
        h0 = self.policy.net.initial_latent(obs)
        loss, losses = self.policy.compute_losses(
            h0, actions, pi_t, v_t, r_t, z_t, loss_weights
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), losses


__all__ = ["LightZeroAdapter"]

