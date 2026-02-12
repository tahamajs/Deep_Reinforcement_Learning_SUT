"""Train CQL on offline dataset."""
import argparse
import os
import numpy as np
import torch
from tqdm import trange

from projects.amasa.amasa.offline.cql import CQLAgent, CQLConfig, make_minibatches
from projects.amasa.amasa.envs.suturing_env import SuturingEnv


def load_buffer(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main(args):
    buffer = load_buffer(args.dataset)
    obs_dim = buffer["obs"].shape[1]
    act_dim = buffer["actions"].shape[1]
    cfg = CQLConfig(obs_dim=obs_dim, act_dim=act_dim, device=args.device, cql_alpha=args.cql_alpha)
    agent = CQLAgent(cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    steps = args.steps
    batch_size = args.batch_size
    n = buffer["obs"].shape[0]
    for step in trange(steps):
        idx = np.random.randint(0, n, size=batch_size)
        batch = (
            torch.as_tensor(buffer["obs"][idx], device=cfg.device),
            torch.as_tensor(buffer["actions"][idx], device=cfg.device),
            torch.as_tensor(buffer["rewards"][idx], device=cfg.device).unsqueeze(-1),
            torch.as_tensor(buffer["next_obs"][idx], device=cfg.device),
            torch.as_tensor(buffer["dones"][idx], device=cfg.device).unsqueeze(-1),
        )
        metrics = agent.update(batch)
        if (step + 1) % args.log_every == 0:
            print({k: round(v, 3) for k, v in metrics.items()})
        if (step + 1) % args.save_every == 0:
            path = os.path.join(args.out_dir, f"cql_step{step+1}.pt")
            agent.save(path)
    agent.save(os.path.join(args.out_dir, "cql_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/amasa_offline.npz")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cql_alpha", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()
    main(args)
