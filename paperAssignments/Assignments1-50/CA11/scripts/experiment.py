#!/usr/bin/env python3
"""
Experiment runner for CA11 (TWM-SSD).
Provides a documented CLI to run quick experiments with logging and checkpointing.
"""
import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb

    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from src.config import get_default_config
from src.data import RandomTrajectoryDataset
from src.model import TWMSSDModel, TWMSSDImageModel
from src.losses import total_model_loss
from src.utils import set_seed, get_device
from src.tokenizer import ImageVQVAE


def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ca11")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run(cfg, steps: int, save_dir: str):
    set_seed(0)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(save_dir)
    logger.info("Starting experiment")
    ds = RandomTrajectoryDataset(seq_len=cfg.seq_len, d_model=cfg.d_model, size=512)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    # Use image tokenizer + image-aware wrapper as optional example
    img_vq = ImageVQVAE(codebook_size=256, d_model=cfg.d_model, in_ch=3)
    backbone = TWMSSDModel(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers
    )
    model = TWMSSDImageModel(img_vq, backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

    if _HAS_WANDB:
        wandb.init(project="ca11_twm_ssd", config=vars(cfg))
        wandb.watch(model, log="all", log_freq=10)

    it = 0
    for epoch in range(1000000):
        for obs, acts in dl:
            obs = obs.to(device)
            acts = acts.to(device)
            pred_obs, pred_reward = model(obs, acts)
            reward_target = torch.zeros_like(pred_reward)
            loss = total_model_loss(pred_obs, obs, pred_reward, reward_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            it += 1
            if it % 10 == 0:
                logger.info(f"iter={it} loss={loss.item():.6f}")
                tb_writer.add_scalar("train/loss", loss.item(), it)
                if _HAS_WANDB:
                    wandb.log({"train/loss": loss.item(), "iter": it})
            if it >= steps:
                # save checkpoint
                ckpt = {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "cfg": vars(cfg),
                    "iter": it,
                }
                torch.save(ckpt, os.path.join(save_dir, f"ckpt_{it}.pt"))
                logger.info(f"Saved checkpoint at iter {it}")
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="runs/ca11")
    args = parser.parse_args()
    cfg = get_default_config()
    run(cfg, steps=args.steps, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
