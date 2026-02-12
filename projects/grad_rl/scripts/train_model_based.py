import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sb3_contrib import TRPO

from grad_rl import make_env, set_seed, evaluate_sb3


# --------------------------- Tabular Dyna-Q ---------------------------------

def dyna_q(env_id, cfg, seed):
    import gymnasium as gym

    env = gym.make(env_id)
    set_seed(seed)
    alpha = cfg.get("alpha", 0.1)
    gamma = cfg.get("gamma", 0.99)
    epsilon = cfg.get("epsilon", 0.1)
    planning_steps = cfg.get("planning_steps", 20)
    model = defaultdict(lambda: defaultdict(list))  # (s,a) -> list of (r, s')
    q = defaultdict(lambda: np.zeros(env.action_space.n))

    episodes = cfg.get("episodes", 2000)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q[obs]))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            best_next = np.max(q[next_obs])
            q[obs][action] += alpha * (reward + gamma * best_next - q[obs][action])
            model[obs][action].append((reward, next_obs))
            # planning updates
            for _ in range(planning_steps):
                s = np.random.choice(list(model.keys()))
                a = np.random.choice(list(model[s].keys()))
                r, s_next = model[s][a][np.random.randint(len(model[s][a]))]
                q[s][a] += alpha * (r + gamma * np.max(q[s_next]) - q[s][a])
            obs = next_obs
        rewards.append(total)
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1} avg reward {np.mean(rewards[-100:]):.2f}")
    return {"episodes": episodes, "avg_last_100": float(np.mean(rewards[-100:]))}


# --------------------------- ME-TRPO style ----------------------------------
class EnsembleModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


def me_trpo(env_id, cfg, seed):
    set_seed(seed)
    env = make_env(env_id, seed=seed, vec=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ensemble_size = cfg.get("ensemble_size", 4)
    horizon = cfg.get("rollout_horizon", 5)
    total_steps = cfg.get("trpo_total_steps", 150_000)

    models = [EnsembleModel(obs_dim, act_dim) for _ in range(ensemble_size)]
    optims = [optim.Adam(m.parameters(), lr=1e-3) for m in models]

    trpo = TRPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=0.95,
        target_kl=0.01,
        tensorboard_log="projects/grad_rl/outputs/tb/model_based",
        verbose=0,
        seed=seed,
    )

    obs, _ = env.reset()
    for step in range(total_steps):
        action, _ = trpo.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # train ensemble one step
        obs_t = torch.tensor(obs, dtype=torch.float32)
        act_t = torch.tensor(action, dtype=torch.float32)
        next_t = torch.tensor(next_obs, dtype=torch.float32)
        for m, opt in zip(models, optims):
            pred = m(obs_t, act_t)
            loss = ((pred - next_t) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        obs = next_obs
        if done:
            obs, _ = env.reset()

        # periodic policy updates using imagination rollouts
        if (step + 1) % 2048 == 0:
            imagined_obs = torch.tensor(np.stack([env.observation_space.sample() for _ in range(256)]), dtype=torch.float32)
            returns = []
            for t in range(horizon):
                with torch.no_grad():
                    actions = torch.randn(256, act_dim)  # random exploratory
                    next_preds = torch.stack([m(imagined_obs, actions) for m in models]).mean(0)
                imagined_obs = next_preds
            # we do not craft rewards; this is a stub to illustrate rollout usage
            trpo.learn(total_timesteps=2048, reset_num_timesteps=False, progress_bar=False)

    metrics = evaluate_sb3(trpo, env, n_eval_episodes=5)
    return trpo, metrics


# --------------------------- CLI --------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Model-based chain: Dyna-Q tabular and ME-TRPO ensemble")
    p.add_argument("--config", default="projects/grad_rl/configs/model_based.yaml")
    p.add_argument("--algo", choices=["dyna-q", "me-trpo"], default=None)
    p.add_argument("--env", default=None)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    algo = args.algo or cfg.get("algo", "dyna-q")
    env_id = args.env or cfg.get("env")

    if algo == "dyna-q":
        metrics = dyna_q(env_id, cfg, args.seed)
        out = Path(cfg.get("save_path", "projects/grad_rl/outputs/model_based/dyna_q.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Saved", out)
        return

    model, metrics = me_trpo(cfg.get("trpo_env", env_id), cfg, args.seed)
    out = Path(cfg.get("save_path", "projects/grad_rl/outputs/model_based/me_trpo.zip"))
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    metrics_path = out.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved", out)
    print("Eval", metrics)


if __name__ == "__main__":
    main()
