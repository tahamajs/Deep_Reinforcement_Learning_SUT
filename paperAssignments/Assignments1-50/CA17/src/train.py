from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from .config import get_default_config
from .model import MLPPolicy
from .losses import policy_gradient_loss, entropy_loss
from .utils import ensure_dir, set_seed, save_checkpoint


def compute_returns(rewards, gamma: float):
    """Compute discounted returns for an episode (simple Monte-Carlo)."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return np.array(returns, dtype=np.float32)


def train_once(config=None):
    cfg = config or get_default_config()
    import gym

    set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(cfg.env_name)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    input_dim = int(np.array(obs).shape[0])
    output_dim = env.action_space.n
    policy = MLPPolicy(
        input_dim=input_dim, output_dim=output_dim, hidden_size=cfg.hidden_size
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    save_dir = ensure_dir(cfg.save_dir)

    total_steps = 0
    while total_steps < cfg.total_timesteps:
        # collect one episode
        observations, actions, rewards, log_probs = [], [], [], []
        done = False
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        while not done and total_steps < cfg.total_timesteps:
            obs_tensor = torch.tensor(
                np.asarray(obs), dtype=torch.float32, device=device
            )
            dist = policy.get_action_dist(obs_tensor)
            action = dist.sample().item()
            lp = dist.log_prob(torch.tensor(action, device=device))
            step = env.step(action)
            if len(step) == 5:
                next_obs, reward, terminated, truncated, info = step
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, info = step
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            log_probs.append(lp)
            obs = next_obs
            total_steps += 1

        returns = compute_returns(rewards, cfg.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        log_probs_tensor = torch.stack(log_probs)
        advantages = returns - returns.mean()
        pg_loss = policy_gradient_loss(log_probs_tensor, advantages)
        ent_loss = entropy_loss(
            policy.forward(
                torch.tensor(
                    np.asarray(observations), dtype=torch.float32, device=device
                )
            )
        )
        loss = pg_loss + ent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Steps: {total_steps}\tEpisode return: {returns.sum():.2f}\tLoss: {loss.item():.4f}"
        )
        # checkpoint
        ckpt_path = save_dir / f"ca17_step_{total_steps}.pt"
        save_checkpoint(
            {
                "model_state": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": total_steps,
            },
            ckpt_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()
    cfg = get_default_config()
    if args.env:
        cfg = type(cfg)(**{**asdict(cfg), "env_name": args.env})
    train_once(cfg)


if __name__ == "__main__":
    main()















