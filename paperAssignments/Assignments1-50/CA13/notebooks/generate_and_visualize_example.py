#!/usr/bin/env python3
\"\"\"Example script showing how to generate branches JSON from planner and call visualization scripts.

This script:
 - constructs dummy RSSM/actor/value (or uses planner.simulate_branches if available)
 - runs simulate_branches to produce branches
 - serializes branches to JSON (tensor -> list)
 - calls the plotting utilities (in-process) to produce PNG figures

Run:
 python notebooks/generate_and_visualize_example.py --out_prefix ../pictures/fig_branch_example.png
\"\"\"
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

try:
    from planner import simulate_branches, CheckpointBuffer
except Exception:
    simulate_branches = None

from planner.simgolf_latent import Branch as BranchCls  # fallback branch container

from analysis.visualize_branches import plot_branch_returns, plot_topk_actions, plot_branch_latent_pca


class DummyRSSM:
    def step(self, z, a):
        # simple deterministic transition
        z_next = (z + 0.1 * (a.reshape(z.shape) if isinstance(a, torch.Tensor) else torch.zeros_like(z))).detach()
        r = 0.1
        gamma = 1.0
        return z_next, r, gamma


class DummyActor:
    def sample(self, z):
        return torch.randn((z.shape[0], 1), device=z.device) * 0.1


def branches_to_json(branches):
    out = []
    for br in branches:
        traj = []
        for (z, a, r, gamma) in br.traj:
            zlist = z.detach().cpu().numpy().tolist() if hasattr(z, "detach") else list(z)
            alist = a.detach().cpu().numpy().tolist() if hasattr(a, "detach") else list(a)
            traj.append([zlist, alist, float(r), float(gamma)])
        out.append({"ret": float(br.ret), "traj": traj})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_prefix", type=str, default="../pictures/fig_branch_example.png")
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--H", type=int, default=12)
    args = p.parse_args()

    device = torch.device("cpu")
    z0 = torch.zeros((1, 8), device=device)
    rssm = DummyRSSM()
    actor = DummyActor()

    class DummyValue:
        def __call__(self, z):
            return torch.zeros(1, device=z.device)

    value_fn = DummyValue()

    # Build a lightweight config object
    class Cfg:
        B = args.B
        H = args.H
        kappa = 0.0
        cem = type("c", (), {"enabled": False})

    cfg = Cfg()

    if simulate_branches is None:
        # fallback: use local simple rollout
        branches = []
        for b in range(cfg.B):
            z = z0.clone()
            traj = []
            ret = 0.0
            disc = 1.0
            for h in range(cfg.H):
                a = actor.sample(z)
                z, r, gamma = rssm.step(z, a)
                traj.append((z.squeeze(0), a.squeeze(0), float(r), float(gamma)))
                ret += disc * float(r)
                disc *= float(gamma)
            branches.append(BranchCls(ret, traj))
    else:
        branches = simulate_branches(rssm, actor, value_fn, z0, cfg)

    json_branches = branches_to_json(branches)
    out_json = Path(args.out_prefix).with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(json_branches, f)
    print(f"Wrote branches JSON to {out_json}")

    # call plotting functions to create figures
    plot_branch_returns(json_branches, str(Path(args.out_prefix).with_suffix("_returns.png")))
    plot_topk_actions(json_branches, str(Path(args.out_prefix).with_suffix("_actions.png")))
    plot_branch_latent_pca(json_branches, str(Path(args.out_prefix).with_suffix("_pca.png")))
    print(f"Wrote example figures with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()









