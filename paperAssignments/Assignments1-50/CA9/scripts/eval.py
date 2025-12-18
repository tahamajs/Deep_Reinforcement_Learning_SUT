"""Evaluation runner for AU-DMG policies (requires gymnasium)."""
import argparse
import os
from statistics import mean

try:
    import gymnasium as gym
except Exception:
    gym = None

from ..src.models.policy import GaussianPolicy
from ..src.config import default_config
from ..src.utils.logger import plot_series
import torch


def evaluate_policy(policy: GaussianPolicy, env_name: str, episodes: int = 5) -> float:
    if gym is None:
        raise RuntimeError("gymnasium not available")
    env = gym.make(env_name)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < 1000:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                a = policy.sample(obs_t).squeeze(0).numpy()
            obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
            steps += 1
        returns.append(ep_ret)
    return mean(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    cfg = default_config()
    # instantiate a fresh policy (user can modify to load weights)
    policy = GaussianPolicy(s_dim=cfg.latent_dim, a_dim=2)
    try:
        avg = evaluate_policy(policy, args.env, episodes=args.episodes)
        print(f"Average return over {args.episodes} episodes: {avg:.3f}")
    except Exception as e:
        print("Evaluation failed:", e)


if __name__ == "__main__":
    main()

