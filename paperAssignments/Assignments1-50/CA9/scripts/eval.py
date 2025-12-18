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


def evaluate_policy(policy: GaussianPolicy, env_name: str, episodes: int = 5):
    if gym is None:
        raise RuntimeError("gymnasium not available")
    env = gym.make(env_name)
    returns = []
    trajectories = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        traj = {"obs": [], "acts": [], "rews": []}
        while not done and steps < 1000:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                a = policy.sample(obs_t).squeeze(0).numpy()
            obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
            traj["obs"].append(obs)
            traj["acts"].append(a)
            traj["rews"].append(float(r))
            steps += 1
        returns.append(ep_ret)
        trajectories.append(traj)
    return mean(returns), trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="antmaze-medium-diverse-v2")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    cfg = default_config()
    try:
        if args.ckpt:
            # load checkpoint and restore AUDMG/policy if available
            import torch
            from ..src.algos.au_dmg import AUDMG

            data = torch.load(args.ckpt, map_location="cpu")
            # infer a_dim from policy state, infer input dim from q state
            policy_state = data.get("policy_state", None)
            q_state = data.get("q_state", None)
            if policy_state is not None and q_state is not None:
                a_dim = policy_state["mu_head.weight"].shape[0]
                # find first q weight to get input dim
                first_q_key = next(
                    k for k in q_state.keys() if "qs.0.net.0.weight" in k
                )
                input_dim = q_state[first_q_key].shape[1]
                s_dim = input_dim - a_dim
                audmg = AUDMG(s_dim=s_dim, a_dim=a_dim, cfg=cfg)
                audmg.load_checkpoint(args.ckpt, map_location="cpu")
                policy = audmg.policy
            else:
                # fallback: instantiate default policy and try to load policy_state
                a_dim = 2
                policy = GaussianPolicy(s_dim=cfg.latent_dim, a_dim=a_dim)
                if policy_state is not None:
                    policy.load_state_dict(policy_state)
        else:
            # instantiate a fresh policy (user can modify to load weights)
            policy = GaussianPolicy(s_dim=cfg.latent_dim, a_dim=2)

        avg, trajectories = evaluate_policy(policy, args.env, episodes=args.episodes)
        print(f"Average return over {args.episodes} episodes: {avg:.3f}")
        # save a simple plot of returns (single point repeated) and placeholder trajectory file
        out_dir = os.path.join("outputs", "ca9", "eval")
        os.makedirs(out_dir, exist_ok=True)
        try:
            from ..src.utils.logger import plot_series

            # save returns plot
            plot_series(
                [0, 1],
                {"returns": [avg, avg]},
                os.path.join(out_dir, "returns.png"),
                title="Eval returns",
            )
            # save trajectories and per-episode reward plots
            import numpy as _np

            for i, traj in enumerate(trajectories):
                ep_dir = os.path.join(out_dir, f"ep_{i}")
                os.makedirs(ep_dir, exist_ok=True)
                _np.save(
                    os.path.join(ep_dir, "obs.npy"),
                    _np.array(traj["obs"], dtype=object),
                )
                _np.save(
                    os.path.join(ep_dir, "acts.npy"),
                    _np.array(traj["acts"], dtype=object),
                )
                _np.save(
                    os.path.join(ep_dir, "rews.npy"),
                    _np.array(traj["rews"], dtype=float),
                )
                # plot rewards over steps
                plot_series(
                    list(range(len(traj["rews"]))),
                    {"rewards": traj["rews"]},
                    os.path.join(ep_dir, "rewards.png"),
                    title=f"Episode {i} rewards",
                )
        except Exception:
            pass
    except Exception as e:
        print("Evaluation failed:", e)


if __name__ == "__main__":
    main()













