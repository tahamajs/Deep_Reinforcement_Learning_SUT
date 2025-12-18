"""Minimal training script for MA-EZV2 (sanity/demo).

This script is intentionally lightweight and safe to run on CPU for small tests.
It does not perform environment rollouts; instead it demonstrates model, loss, and optimizer integration.
"""

import argparse
import yaml
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..policy.efficientzero_v2_ma import EfficientZeroV2Policy


def make_fake_dataset(obs_dim: int, joint_action_dim: int, n: int = 1024):
    obs = torch.randn(n, obs_dim)
    # actions represented as dense vectors for dynamics head (one-hot-like)
    actions = torch.randn(n, joint_action_dim)
    # soft targets for policy (visit counts)
    pi_targets = torch.softmax(torch.randn(n, joint_action_dim), dim=-1)
    value_targets = torch.randn(n)
    reward_targets = torch.randn(n)
    prefix_targets = torch.randn(n)
    return TensorDataset(
        obs, actions, pi_targets, value_targets, reward_targets, prefix_targets
    )


def train(cfg_path: str, device: str = "cpu"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    loss_weights = cfg.get("loss_weights", {})

    policy = EfficientZeroV2Policy(
        obs_dim=mcfg["obs_dim"],
        latent_dim=mcfg["latent_dim"],
        joint_action_dim=mcfg["joint_action_dim"],
        per_agent_action_dims=mcfg.get("per_agent_action_dims"),
        device=device,
    )

    ds = make_fake_dataset(mcfg["obs_dim"], mcfg["joint_action_dim"], n=512)
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True)
    optim = torch.optim.Adam(policy.parameters(), lr=tcfg["lr"])

    policy.train()
    for epoch in range(tcfg["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            obs, actions, pi_t, v_t, r_t, z_t = [b.to(device) for b in batch]
            # encode
            h0 = policy.net.initial_latent(obs)
            loss, losses = policy.compute_losses(
                h0, actions, pi_t, v_t, r_t, z_t, loss_weights
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optim.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{tcfg['epochs']} avg_loss={epoch_loss/len(loader):.4f}")

    # save checkpoint
    out_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "ma_ezv2_ckpt.pt")
    policy.save(ckpt)
    print("Saved checkpoint:", ckpt)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../configs/ma_ezv2_default.yaml"
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    cfg_path = os.path.abspath(args.config)
    train(cfg_path, device=args.device)


if __name__ == "__main__":
    cli()







