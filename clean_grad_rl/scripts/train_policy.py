import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from grad_rl import make_env, set_seed, evaluate_sb3


def parse_args():
    p = argparse.ArgumentParser(description="Policy-gradient chain: REINFORCE / PPO / TRPO / CPO-lite")
    p.add_argument("--config", default="projects/grad_rl/configs/policy.yaml")
    p.add_argument("--algo", choices=["reinforce", "ppo", "trpo", "cpo"], default=None)
    p.add_argument("--env", default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


class ReinforceAgent(nn.Module):
    def __init__(self, obs_space, action_space, hidden=128):
        super().__init__()
        self.obs_dim = obs_space.shape[0]
        self.n_actions = action_space.n
        self.policy = nn.Sequential(
            nn.Linear(self.obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n_actions)
        )

    def forward(self, x):
        logits = self.policy(x)
        return torch.distributions.Categorical(logits=logits)


def run_reinforce(env_id, total_steps, seed):
    env = make_env(env_id, seed=seed, vec=False)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ReinforceAgent(env.observation_space, env.action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    gamma = 0.99

    all_rewards = []
    steps = 0
    while steps < total_steps:
        log_probs = []
        rewards = []
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = agent(obs_t)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
            steps += 1
            if steps >= total_steps:
                break
        # returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        log_probs_t = torch.stack(log_probs)
        # normalize returns to reduce variance
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = -(log_probs_t * returns_t).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_rewards.append(sum(rewards))
        if len(all_rewards) % 20 == 0:
            print(f"Episode {len(all_rewards)} mean reward {np.mean(all_rewards[-20:]):.2f}")
    env.close()
    return {"episodes": len(all_rewards), "mean_reward": float(np.mean(all_rewards[-20:]))}


def run_sb3(algo, env_id, cfg, total_steps, seed):
    num_envs = int(cfg.get("num_envs", 8))
    set_seed(seed)
    env = SubprocVecEnv([lambda r=i: make_env(env_id, seed=seed + i, vec=False) for i in range(num_envs)])
    policy = cfg.get("policy", "MlpPolicy")

    if algo == "ppo":
        model = PPO(
            policy,
            env,
            learning_rate=cfg.get("learning_rate", 3e-4),
            n_steps=cfg.get("n_steps", 2048 // num_envs),
            batch_size=cfg.get("batch_size", 64),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("lam", 0.95),
            clip_range=cfg.get("clip_range", 0.2),
            tensorboard_log="projects/grad_rl/outputs/tb/policy",
            verbose=1,
            seed=seed,
        )
    elif algo == "trpo":
        model = TRPO(
            policy,
            env,
            learning_rate=cfg.get("learning_rate", 3e-4),
            batch_size=cfg.get("batch_size", 256),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("lam", 0.95),
            target_kl=cfg.get("kl_target", 0.01),
            tensorboard_log="projects/grad_rl/outputs/tb/policy",
            verbose=1,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported algo {algo}")

    model.learn(total_timesteps=total_steps, callback=ProgressBarCallback())
    metrics = evaluate_sb3(model, env, n_eval_episodes=5)
    return model, metrics


class CPOLagrangian:
    """Simplified CPO-like update using PPO + dual variable on cost."""

    def __init__(self, env_id, cfg, seed):
        self.env = SubprocVecEnv([lambda r=i: make_env(env_id, seed=seed + i, vec=False) for i in range(int(cfg.get("num_envs", 8)))])
        self.policy = cfg.get("policy", "MlpPolicy")
        self.cost_limit = cfg.get("cpo_cost_limit", 0.02)
        self.lambda_lr = cfg.get("cpo_lambda_lr", 0.05)
        self.lam = 1.0
        self.ppo = PPO(
            self.policy,
            self.env,
            learning_rate=cfg.get("learning_rate", 3e-4),
            n_steps=cfg.get("n_steps", 128),
            batch_size=cfg.get("batch_size", 64),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("lam", 0.95),
            clip_range=cfg.get("clip_range", 0.2),
            tensorboard_log="projects/grad_rl/outputs/tb/policy",
            verbose=0,
            seed=seed,
        )

    def learn(self, total_steps):
        steps = 0
        while steps < total_steps:
            rollout = self.ppo.collect_rollouts(self.env, self.ppo.rollout_buffer, self.ppo.n_rollout_steps)
            steps += self.ppo.n_rollout_steps * self.env.num_envs
            # compute average cost from infos
            costs = []
            for info in self.ppo.rollout_buffer.infos:
                if isinstance(info, (list, tuple)):
                    for i in info:
                        if isinstance(i, dict) and "cost" in i:
                            costs.append(i["cost"])
                elif isinstance(info, dict) and "cost" in info:
                    costs.append(info["cost"])
            mean_cost = float(np.mean(costs)) if costs else 0.0
            if mean_cost > self.cost_limit:
                self.lam += self.lambda_lr * (mean_cost - self.cost_limit)
            # set additional penalty (used inside PPO loss)
            self.ppo.ent_coef = float(self.lam)
            self.ppo.train()
            if steps % 5000 == 0:
                print(f"steps={steps} mean_cost={mean_cost:.3f} lambda={self.lam:.3f}")
        return self

    def save(self, path):
        self.ppo.save(path)

    def eval(self, episodes=5):
        return evaluate_sb3(self.ppo, self.env, n_eval_episodes=episodes)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    algo = args.algo or cfg.get("algo", "ppo")
    env_id = args.env or cfg.get("env", "CartPole-v1")
    total_steps = args.total_steps or cfg.get("total_steps", 200_000)

    Path("projects/grad_rl/outputs", "tb", "policy").mkdir(parents=True, exist_ok=True)

    if algo == "reinforce":
        metrics = run_reinforce(env_id, total_steps, args.seed)
        out_path = Path(cfg.get("save_path", "projects/grad_rl/outputs/policy/reinforce.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Saved metrics", metrics)
        return

    if algo == "cpo":
        agent = CPOLagrangian(env_id, cfg, args.seed)
        agent.learn(total_steps)
        out = Path(cfg.get("save_path", "projects/grad_rl/outputs/policy/cpo.zip"))
        out.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(out))
        metrics = agent.eval()
        metrics["algo"] = "cpo-lite"
        metrics_path = out.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Saved", out)
        print("Eval", metrics)
        return

    model, metrics = run_sb3(algo, env_id, cfg, total_steps, args.seed)
    out_path = Path(cfg.get("save_path", "projects/grad_rl/outputs/policy/model.zip"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    metrics["algo"] = algo
    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model to", out_path)
    print("Eval:", metrics)


if __name__ == "__main__":
    main()
