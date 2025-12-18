"""
Lightweight training loop for CA11 (TWM-SSD).
This script is intentionally small and import-safe — suitable for quick smoke tests.
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.data import RandomTrajectoryDataset
from src.model import TWMSSDModel
from src.losses import total_model_loss
from src.utils import set_seed, get_device


def train_loop(cfg, steps: int = 10, save_path: str = "ca11_ckpt.pt"):
    set_seed(0)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RandomTrajectoryDataset(seq_len=cfg.seq_len, d_model=cfg.d_model, size=256)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = TWMSSDModel(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    it = 0
    start = time.time()
    while it < steps:
        for obs, acts in dl:
            obs = obs.to(device)
            acts = acts.to(device)
            pred_obs, pred_reward = model(obs, acts)
            # synthetic reward target (zeros) for smoke test
            reward_target = torch.zeros_like(pred_reward)
            loss = total_model_loss(pred_obs, obs, pred_reward, reward_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            it += 1
            if it % 10 == 0 or it >= steps:
                elapsed = time.time() - start
                print(f"iter={it} loss={loss.item():.4f} elapsed={elapsed:.2f}s")
            if it >= steps:
                break

    torch.save({"model_state": model.state_dict(), "cfg": vars(cfg)}, save_path)
    print("Saved checkpoint to", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save", type=str, default="ca11_ckpt.pt")
    args = parser.parse_args()
    cfg = get_default_config()
    train_loop(cfg, steps=args.steps, save_path=args.save)


if __name__ == "__main__":
    main()













