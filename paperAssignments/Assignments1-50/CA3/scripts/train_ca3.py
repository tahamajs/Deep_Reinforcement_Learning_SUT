"""Non-interactive training script for CA3 REINFORCE demo.

Run from repository root:
python paperAssignments/Assignments1-50/CA3/scripts/train_ca3.py

This script runs a small number of episodes (default=20) suitable for CI or smoke tests.
"""

import argparse
import os
import torch
import gym
from src.config import Config
from src.model import MLPPolicy
from src.data import collect_episode
from src.utils import set_seed, returns_to_tensor, discounted_returns, ensure_dir
from src.losses import reinforce_loss


def main(args):
    cfg = Config()
    cfg.max_episodes = args.episodes
    set_seed(cfg.seed)
    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = MLPPolicy(obs_dim, action_dim, hidden_sizes=cfg.hidden_sizes)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    returns_history = []
    for ep in range(cfg.max_episodes):
        ep_data = collect_episode(
            env, policy, device="cpu", max_steps=cfg.max_steps_per_episode
        )
        rewards = ep_data["rewards"]
        log_probs = torch.as_tensor(ep_data["log_probs"], dtype=torch.float32)
        G = discounted_returns(rewards, cfg.gamma)
        G_tensor = returns_to_tensor(G)
        loss = reinforce_loss(log_probs, G_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        returns_history.append(sum(rewards))
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}\tReturn: {returns_history[-1]:.2f}")

    # save a lightweight checkpoint
    ensure_dir(cfg.save_dir)
    ckpt_path = os.path.join(cfg.save_dir, "checkpoint_smoke.pth")
    torch.save(
        {"policy_state": policy.state_dict(), "returns": returns_history}, ckpt_path
    )
    print("Saved checkpoint to", ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of episodes to run"
    )
    args = parser.parse_args()
    main(args)











