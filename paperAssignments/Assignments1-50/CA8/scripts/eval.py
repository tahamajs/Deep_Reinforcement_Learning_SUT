"""
Evaluation script for MaxSink: run an agent for a given number of episodes and report mean/std returns.
"""

import argparse
import os
import sys
import numpy as np

# make local src importable
THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import cfg  # type: ignore
from envs import ToTheMaxWrapper  # type: ignore
from agent import MaxSinkAgent  # type: ignore

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default=cfg.env_name)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--beta", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    try:
        import gymnasium as gym
    except Exception:
        print("gymnasium not available; install required env packages.")
        return

    beta = cfg.beta if args.beta is None else args.beta
    env = ToTheMaxWrapper(gym.make(args.env), beta=beta)

    res = env.reset()
    if isinstance(res, tuple):
        obs, info = res
    else:
        obs = res
    state_dim = np.asarray(obs).ravel().shape[0]
    n_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]

    agent = MaxSinkAgent(state_dim=state_dim, n_actions=int(n_actions), device=str(cfg.device))

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        try:
            sd = torch.load(args.checkpoint, map_location=cfg.device)
            if "critic_state_dict" in sd:
                agent.critic.load_state_dict(sd["critic_state_dict"])
                print("Loaded critic weights from checkpoint.")
            else:
                agent.critic.load_state_dict(sd)
                print("Loaded state dict from checkpoint (assumed critic weights).")
        except Exception as e:
            print("Could not load checkpoint:", e)

    returns = []
    for ep in range(args.episodes):
        res = env.reset()
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs = res
        done = False
        ep_return = 0.0
        while not done:
            s = torch.from_numpy(np.asarray(obs).ravel()).float()
            a = agent.act(s)
            out = env.step(a)
            if len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = terminated or truncated
            else:
                obs, r, done, info = out
            ep_return += float(r)
        returns.append(ep_return)
        print(f"Episode {ep+1:02d}: return={ep_return:.3f}")

    returns = np.array(returns)
    print("--- Summary ---")
    print(f"episodes={len(returns)} mean={returns.mean():.3f} std={returns.std():.3f}")


if __name__ == "__main__":
    main()
