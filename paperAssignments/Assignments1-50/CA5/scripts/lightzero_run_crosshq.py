"""Run a LightZero-style planning demo using CrossHQ actor+critic wired into MuZero adapter.

Produces a visualization of priors, visit counts and estimated Q-values for the root action set.
Enhanced plots: priors, visits, estimated Q per action, and BN running stats where available.
"""

import argparse
import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.crosshq.model import CrossQCritic, GaussianPolicy
from src.crosshq.mcts_adapter import CrossHQMCTSAdapter
from src.mcts.puct import PUCT


def make_action_set(action_dim: int, n: int = 5):
    # simple radial action candidates around zero
    actions = []
    for i in range(n):
        vec = torch.zeros(action_dim)
        vec[i % action_dim] = (i // action_dim) - (n // 2)
        actions.append(vec)
    return actions


def run_crosshq_planning(
    seed: int = 0, sims: int = 200, horizon: int = 3, out_dir: str = "pictures"
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    s_dim = 4
    a_dim = 2
    # instantiate CrossHQ modules (untrained random weights for demo)
    critic = CrossQCritic(state_dim=s_dim, action_dim=a_dim, hidden_dim=128, depth=2)
    actor = GaussianPolicy(obs_dim=s_dim, action_dim=a_dim, hidden=128, depth=2)

    action_set = make_action_set(a_dim, n=6)

    adapter = CrossHQMCTSAdapter(critic, actor, action_set)
    puct = PUCT(
        adapter,
        action_space=list(range(len(action_set))),
        c_puct=1.0,
        dirichlet_alpha=0.25,
    )

    state = torch.zeros(s_dim)  # demo initial state

    for t in range(horizon):
        root = puct.search(state, num_simulations=sims)

        # collect stats
        actions = sorted(root.children.keys())
        priors = [root.children[a].prior for a in actions]
        visits = [root.children[a].visits for a in actions]
        # estimate Q via adapter per action (simulate state with that action appended)
        q_est = [adapter.value(state) for _ in actions]

        # combined plot: priors (bar), visits (line), q_est (line)
        x = list(map(str, actions))
        fig, (ax_top, ax_bot) = plt.subplots(
            nrows=2, ncols=1, figsize=(7, 6), gridspec_kw={"height_ratios": [2, 1]}
        )

        ax_top.bar(x, priors, alpha=0.7, label="priors")
        ax_top.set_ylabel("prior")
        ax_top.set_title(f"CrossHQ MCTS root stats t={t}")
        ax_top_twin = ax_top.twinx()
        ax_top_twin.plot(x, visits, color="C1", marker="o", label="visits")
        ax_top_twin.set_ylabel("visits")

        ax_bot.plot(x, q_est, color="C2", marker="s", label="estimated Q")
        ax_bot.set_ylabel("estimated Q")
        ax_bot.set_xlabel("action index")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"crosshq_root_t{t}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved root visualization to {out_path}")

        # BN running stats plot (if critic has BatchNorm layers)
        try:
            bn_means = []
            bn_vars = []
            for m in critic.q1.net.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    bn_means.append(m.running_mean.detach().cpu().abs().mean().item())
                    bn_vars.append(m.running_var.detach().cpu().mean().item())
            if bn_means:
                fig2, ax2 = plt.subplots(figsize=(7, 2.5))
                idx = list(range(len(bn_means)))
                ax2.plot(idx, bn_means, marker="o", label="mean_abs")
                ax2.plot(idx, bn_vars, marker="s", label="var")
                ax2.set_xlabel("BN layer index")
                ax2.set_title(f"BN running stats t={t}")
                ax2.legend()
                out_bn = os.path.join(out_dir, f"crosshq_bn_t{t}.png")
                fig2.tight_layout()
                fig2.savefig(out_bn, dpi=200)
                plt.close(fig2)
                print(f"Saved BN stats visualization to {out_bn}")
        except Exception:
            pass

        # pick best action by visits and advance state heuristically
        best = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        # simple deterministic state update: add unit vector for chosen action index
        state = state.clone()
        state[best % s_dim] += 1.0

    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--out", type=str, default="pictures")
    args = p.parse_args()
    run_crosshq_planning(
        seed=args.seed, sims=args.sims, horizon=args.horizon, out_dir=args.out
    )












