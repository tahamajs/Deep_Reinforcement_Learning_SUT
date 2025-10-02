"""
DQN Training Script for Atari Environments

This script trains a DQN agent on Atari games.

Usage:
    python run_dqn_atari.py [options]

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import argparse
import os
import sys
import time
import gym
import numpy as np
import tensorflow as tf
from collections import namedtuple
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.dqn import DQNAgent
from dqn_utils import LinearSchedule, PiecewiseSchedule
import logz
from atari_wrappers import wrap_deepmind
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])
def atari_learn(env_name, num_timesteps, seed=0, double_q=True):
    """Train DQN on Atari environment."""

    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    env = wrap_deepmind(env)
    exploration = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
        ],
        outside_value=0.1,
    )
    lr_schedule = PiecewiseSchedule(
        [
            (0, 1e-4),
            (2e6, 5e-5),
        ],
        outside_value=5e-5,
    )
    optimizer = tf.train.AdamOptimizer
    optimizer_spec = OptimizerSpec(
        constructor=optimizer, kwargs=dict(), lr_schedule=lr_schedule
    )
    agent = DQNAgent(
        env=env,
        optimizer_spec=optimizer_spec,
        session=None,
        exploration=exploration,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=double_q,
    )
    agent.sess = tf.Session()
    agent.sess.run(tf.global_variables_initializer())
    start_time = time.time()
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    agent.replay_buffer_idx = agent.replay_buffer.store_frame(obs)

    for t in range(num_timesteps):

        agent.step_env()
        agent.update_model()
        if t % 10000 == 0:
            print(f"Timestep {t}")
            if len(episode_rewards) > 0:
                print(f"Mean reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
                print(f"Episodes: {len(episode_rewards)}")
                print(f"Exploration: {exploration.value(t):.3f}")
                print(f"Learning rate: {lr_schedule.value(t):.6f}")
                print(f"Time elapsed: {(time.time() - start_time) / 60:.1f} minutes")
        if hasattr(env, "get_episode_rewards"):
            current_rewards = env.get_episode_rewards()
            if len(current_rewards) > len(episode_rewards):
                episode_rewards = current_rewards
                episode_lengths = env.get_episode_lengths()

    print("Training completed!")
    return episode_rewards, episode_lengths
def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="Atari environment name")
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=int(2e6),
        help="Number of timesteps to train",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--double_q", action="store_true", default=True, help="Use double Q-learning"
    )
    args = parser.parse_args()
    rewards, lengths = atari_learn(
        env_name=args.env_name,
        num_timesteps=args.num_timesteps,
        seed=args.seed,
        double_q=args.double_q,
    )

    print(f"Final mean reward: {np.mean(rewards[-100:]):.2f}")
if __name__ == "__main__":
    main()