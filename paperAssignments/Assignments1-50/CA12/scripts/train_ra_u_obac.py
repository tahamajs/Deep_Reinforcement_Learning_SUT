"""Minimal training script for RA-U-OBAC (guarded main)."""

from argparse import ArgumentParser
import random
import time

import gym
import numpy as np
import torch

from ..src.config import Config
from ..src.agent import RAUOBACAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_demo(cfg: Config, steps: int = 1000):
    set_seed(cfg.seed)
    env = gym.make("Hopper-v2")
    # handle gym / gymnasium API differences for reset/step returning tuples
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = RAUOBACAgent(state_dim, action_dim, cfg)

    obs = obs
    episode_states = []
    episode_actions = []
    episode_rewards = []
    total_steps = 0
    start = time.time()
    while total_steps < steps:
        action = env.action_space.sample()
        step_ret = env.step(action)
        # support (obs, reward, done, info) and (obs, reward, terminated, truncated, info)
        if len(step_ret) == 4:
            next_obs, reward, done, _ = step_ret
        else:
            next_obs, reward, terminated, truncated, _ = step_ret
            done = terminated or truncated
        episode_states.append(torch.tensor(obs, dtype=torch.float32))
        episode_actions.append(torch.tensor(action, dtype=torch.float32))
        episode_rewards.append(torch.tensor(float(reward), dtype=torch.float32))
        obs = next_obs
        total_steps += 1
        if done or len(episode_rewards) >= 200:
            states = torch.stack(episode_states)
            actions = torch.stack(episode_actions)
            rewards = torch.stack(episode_rewards)
            agent.retrieval_buffer.add_trajectory(
                states, actions, rewards, gamma=cfg.gamma
            )
            episode_states, episode_actions, episode_rewards = [], [], []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

        # periodic updates (very small, demo-only)
        if agent.retrieval_buffer.size >= 32 and total_steps % 10 == 0:
            s_batch, a_batch, rtg_batch = agent.retrieval_buffer.sample_batch(
                min(cfg.batch_size, 32)
            )
            agent.update_critic(s_batch, a_batch, rtg_batch.squeeze(1))
            agent.update_online_actor(s_batch)

    elapsed = time.time() - start
    print(f"Demo finished: {total_steps} env steps in {elapsed:.2f}s")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()
    cfg = Config()
    run_demo(cfg, steps=args.steps)
