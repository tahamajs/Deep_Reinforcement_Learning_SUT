"""Demo script that runs a single-policy inference and MCTS search to produce a policy."""

import os
import yaml
import torch

from paperAssignments.Assignments1_50.CA10.policy.efficientzero_v2_ma import (
    EfficientZeroV2Policy,
)
from paperAssignments.Assignments1_50.CA10.mcts.search import MCTS


def demo(config_path: str = None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "../configs/ma_ezv2_default.yaml"
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    device = "cpu"
    policy = EfficientZeroV2Policy(
        obs_dim=mcfg["obs_dim"],
        latent_dim=mcfg["latent_dim"],
        joint_action_dim=mcfg["joint_action_dim"],
        per_agent_action_dims=mcfg.get("per_agent_action_dims"),
        device=device,
    )

    obs = torch.randn(1, mcfg["obs_dim"])
    logits_joint, logits_agents, value, prefix = policy.initial_infer(obs)
    print("Root value:", value)
    mcts = MCTS(policy.net, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_frac=0.25)
    visits, policy_from_visits = mcts.run(
        policy.net.initial_latent(obs), num_simulations=30, topk=6
    )
    print("Visit counts (nonzero):", visits[visits > 0])
    print("Policy from visits (sum):", policy_from_visits.sum().item())


if __name__ == "__main__":
    demo()

