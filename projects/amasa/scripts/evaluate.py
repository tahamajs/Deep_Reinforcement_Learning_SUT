"""Evaluate checkpoints on reward/safety and plot Pareto frontier."""
import argparse
import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from projects.amasa.amasa.envs.suturing_env import SuturingEnv
from projects.amasa.amasa.offline.cql import CQLAgent, CQLConfig
from projects.amasa.amasa.safety.shield import SafetyShield


def rollout(agent, env, episodes, shield=None):
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        total_cost = 0.0
        steps = 0
        while not done:
            act = agent.act(obs)
            if shield and shield.trained:
                act, _ = shield.filter(obs, act)
            obs, r, term, trunc, info = env.step(act)
            total_r += r
            total_cost += info.get("cost", 0.0)
            steps += 1
            done = term or trunc
        results.append((total_r, total_cost / max(1, steps)))
    return np.array(results)


def main(args):
    ckpts = sorted(glob.glob(os.path.join(args.checkpoints, "*.pt")))
    if not ckpts:
        raise SystemExit("no checkpoints found")
    env = SuturingEnv(max_steps=args.max_steps, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    points = []
    for ck in ckpts:
        try:
            data = torch.load(ck, map_location=args.device, weights_only=False)
            if isinstance(data, dict) and "actor" in data:
                cfg = data.get("cfg", CQLConfig(obs_dim=obs_dim, act_dim=act_dim, device=args.device))
                agent = CQLAgent(cfg)
                agent.load(ck, map_location=args.device)
            else:
                # likely a HIRO checkpoint; skip
                print(f"skip non-CQL checkpoint {ck}")
                continue
        except Exception as exc:
            print(f"failed to load {ck}: {exc}")
            continue

        shield = None
        shield_path = ck.replace(".pt", ".joblib")
        if os.path.exists(shield_path):
            try:
                shield = SafetyShield.load(shield_path)
            except Exception:
                shield = None

        res = rollout(agent, env, args.episodes, shield)
        mean_r, mean_c = res[:, 0].mean(), res[:, 1].mean()
        points.append((mean_c, mean_r, os.path.basename(ck)))
        print(f"{ck}: reward {mean_r:.1f}, cost {mean_c:.3f}")

    if not points:
        print("no valid CQL checkpoints to evaluate")
        return

    points = np.array(points, dtype=object)
    costs = points[:, 0].astype(float)
    rewards = points[:, 1].astype(float)

    plt.figure(figsize=(6, 4))
    plt.scatter(costs, rewards, c="tab:blue")
    for c, r, name in points:
        plt.annotate(name.replace(".pt", ""), (c, r))
    plt.xlabel("Average cost per step")
    plt.ylabel("Episode reward")
    plt.title("Reward-Safety Pareto")
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    plt.tight_layout()
    plt.savefig(args.out)
    print("Saved plot to", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="plots/pareto.png")
    args = parser.parse_args()
    main(args)
