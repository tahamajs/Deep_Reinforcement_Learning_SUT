"""Minimal training script for RA-U-OBAC (guarded main)."""

from argparse import ArgumentParser
import random
import time

import gym
import numpy as np
import torch
import os
from typing import Dict

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
    # checkpoint dir
    ckpt_dir = os.path.join("outputs", "ca12_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_returns = []
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
            # periodic evaluation & checkpointing
            if total_steps % cfg.save_interval == 0:
                # save checkpoint (actor + critic + offline)
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_{total_steps}.pt")
                torch.save(
                    {
                        "actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict(),
                        "offline_actor": agent.offline_actor.state_dict(),
                        "cfg": cfg.__dict__,
                        "steps": total_steps,
                    },
                    ckpt_path,
                )
                # quick eval: run a few episodes with current actor (deterministic mean)
                eval_env = gym.make(cfg.env_name)
                returns = []
                for _ in range(3):
                    o = eval_env.reset()
                    if isinstance(o, tuple):
                        o = o[0]
                    done = False
                    ret = 0.0
                    while not done:
                        with torch.no_grad():
                            s_t = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
                            mean, _ = agent.actor(s_t)
                            action = torch.tanh(mean).squeeze(0).numpy()
                        step_ret = eval_env.step(action)
                        if len(step_ret) == 4:
                            o, r, done, _ = step_ret
                        else:
                            o, r, term, trunc, _ = step_ret
                            done = term or trunc
                        ret += float(r)
                    returns.append(ret)
                avg_ret = float(np.mean(returns))
                eval_returns.append((total_steps, avg_ret))
                print(
                    f"[Eval] steps={total_steps} avg_return={avg_ret:.2f} saved_ckpt={ckpt_path}"
                )

    elapsed = time.time() - start
    print(f"Demo finished: {total_steps} env steps in {elapsed:.2f}s")
    # write eval returns to file
    if len(eval_returns) > 0:
        out_path = os.path.join(ckpt_dir, "eval_returns.csv")
        with open(out_path, "w") as fh:
            fh.write("steps,avg_return\n")
            for s, r in eval_returns:
                fh.write(f"{s},{r}\n")
        print(f"Saved eval returns to {out_path}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()
    cfg = Config()
    run_demo(cfg, steps=args.steps)
