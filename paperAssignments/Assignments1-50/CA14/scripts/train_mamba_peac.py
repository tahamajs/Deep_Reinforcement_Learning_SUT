"""Training script skeleton for MAMBA-PEAC assignment.

This script is intentionally lightweight and import-safe. It builds model
instances from config, prints summaries and exits. The user can expand it
to run full training loops; by default it performs no heavy computation.
"""
from __future__ import annotations

import argparse
import yaml
import os
from pathlib import Path

import torch

from mamba_core.morph_encoder import MorphEncoder
from mamba_core.world_model import WorldModel
from mamba_core.actor import Actor
from mamba_core.value import ValueNet


def build_models(cfg: dict, obs_dim: int, act_dim: int):
    morph_dim = cfg.get("morph", {}).get("latent_dim", 16)
    wm_latent = cfg.get("world_model", {}).get("latent_dim", 64)
    wm = WorldModel(obs_dim=obs_dim, act_dim=act_dim, stoch_dim=wm_latent, morph_dim=morph_dim)
    morph = MorphEncoder(obs_dim=obs_dim, act_dim=act_dim, latent_dim=morph_dim)
    actor = Actor(latent_dim=wm_latent, morph_dim=morph_dim, act_dim=act_dim)
    value = ValueNet(latent_dim=wm_latent, morph_dim=morph_dim)
    return wm, morph, actor, value


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/mamba_peac.yaml")
    p.add_argument("--env", type=str, default="Walker2d-v4")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--train_morphs", nargs="+", default=["walker2d-v4", "hopper-v4", "halfcheetah-v4"])
    p.add_argument("--heldout_morph", type=str, default="ant-v4")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config {cfg_path} not found. Using defaults from README skeleton.")
        cfg = {}
    else:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

    # For safety, use small dummy dims; user should replace with env observation/action dims
    obs_dim = 24
    act_dim = 6

    wm, morph, actor, value = build_models(cfg, obs_dim, act_dim)

    print("Built models:")
    print(f" WorldModel RSSM deterministic dim: {wm.rssm.deter_dim}, stoch dim: {wm.rssm.stoch_dim}")
    print(f" MorphEncoder latent_dim: {morph.mu.out_features if hasattr(morph.mu, 'out_features') else 'unknown'}")
    print(f" Actor act dim (mu head): {actor.mu.out_features}")
    print(f" Value output dim: 1")

    print("Config summary:")
    print(cfg)

    print("Script is a skeleton. Implement training loop to run experiments as needed.")


if __name__ == "__main__":
    main()
