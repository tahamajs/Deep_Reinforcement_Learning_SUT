from pathlib import Path
import sys
from typing import List, Dict, Any
import time

# ensure local src directory is importable when run from parent
sys.path.append(str(Path(__file__).resolve().parent))

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import CAConfig
from model import ActorCriticEnsemble
from data import ReplayBuffer
from losses import critic_loss, actor_loss, value_ensemble_variance
from utils import set_seed, save_checkpoint


def run_training(cfg: CAConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = gym.make(cfg.env_name)
    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim

    model = ActorCriticEnsemble(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg.hidden_dim,
        ensemble_size=cfg.ensemble_size,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rb = ReplayBuffer(capacity=10000)

    rewards: List[float] = []
    losses: List[float] = []

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    episode_reward = 0.0
    episode_len = 0
    total_steps = 0

    while total_steps < cfg.total_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_t = obs_t.to(device)
        with torch.no_grad():
            logits, values = model.forward(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(int(action))
        done = bool(terminated or truncated)
        rb.add(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor([action]),
            float(reward),
            torch.tensor(next_obs, dtype=torch.float32),
            done,
        )

        episode_reward += reward
        episode_len += 1
        total_steps += 1
        obs = next_obs

        if done or episode_len >= cfg.max_steps_per_episode:
            rewards.append(episode_reward)
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_len = 0

        # update when we have enough samples
        if len(rb) >= cfg.batch_size:
            batch = rb.sample(cfg.batch_size)
            # move tensors
            obs_b = batch["obs"].to(device)
            actions_b = batch["actions"].to(device)
            rewards_b = batch["rewards"].to(device)
            next_obs_b = batch["next_obs"].to(device)
            dones_b = batch["dones"].to(device)

            logits_b, values_b = model.forward(obs_b)
            with torch.no_grad():
                _, next_values_b = model.forward(next_obs_b)
                next_mean = next_values_b.mean(0)
                targets = rewards_b + cfg.gamma * next_mean * (1.0 - dones_b)

            # critic
            c_loss = critic_loss(values_b, targets)
            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()

            # actor
            logits_b, values_b = model.forward(obs_b)
            mean_v = values_b.mean(0)
            advantages = (targets - mean_v).detach()
            var = value_ensemble_variance(values_b).to(device)
            a_loss = actor_loss(logits_b, actions_b, advantages, var, beta=cfg.beta)

            optimizer.zero_grad()
            a_loss.backward()
            optimizer.step()

            losses.append((c_loss.item(), a_loss.item()))

        # occasional checkpoint
        if total_steps % 1000 == 0:
            path = Path.cwd() / "outputs" / f"checkpoint_step_{total_steps}.pt"
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": total_steps,
                },
                str(path),
            )

    env.close()

    # plotting
    Path.cwd().joinpath("pictures").mkdir(exist_ok=True)
    fig_path = Path.cwd().joinpath("pictures", "fig_01_reward.png")
    plt.figure(figsize=(6, 4))
    if len(rewards) > 0:
        plt.plot(np.arange(len(rewards)), np.array(rewards))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Training Returns")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

    return {
        "rewards": rewards,
        "losses": losses,
        "steps": total_steps,
        "checkpoint": str(path) if "path" in locals() else "",
    }


if __name__ == "__main__":
    cfg = CAConfig()
    start = time.time()
    out = run_training(cfg)
    print("Done", out["steps"], "steps, total episodes", len(out["rewards"]))


