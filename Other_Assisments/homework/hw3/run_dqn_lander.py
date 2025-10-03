"""
DQN Training Script for LunarLander

This script trains a DQN agent on the LunarLander environment.

Usage:
    python run_dqn_lander.py [options]

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import argparse
import os
import sys
import time
import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import namedtuple
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.dqn import DQNAgent
from dqn_utils import LinearSchedule, PiecewiseSchedule, ConstantSchedule
import logz
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])
def lander_learn(env_name, num_timesteps, seed=0):
    """Train DQN on LunarLander environment."""

    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    exploration = PiecewiseSchedule(
        [
            (0, 1.0),
            (num_timesteps * 0.1, 0.02),
        ],
        outside_value=0.02,
    )
    lr_schedule = ConstantSchedule(1e-3)
    optimizer = tf.train.AdamOptimizer
    optimizer_spec = OptimizerSpec(
        constructor=optimizer, kwargs=dict(), lr_schedule=lr_schedule
    )
    agent = DQNAgent(
        env=env,
        optimizer_spec=optimizer_spec,
        session=None,
        exploration=exploration,
        replay_buffer_size=50000,
        batch_size=32,
        gamma=1.0,
        learning_starts=1000,
        learning_freq=1,
        frame_history_len=1,
        target_update_freq=1000,
        grad_norm_clipping=10,
        double_q=False,
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
        if t % 1000 == 0:
            print(f"Timestep {t}")
            if len(episode_rewards) > 0:
                print(f"Mean reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
                print(f"Episodes: {len(episode_rewards)}")
                print(f"Exploration: {exploration.value(t):.3f}")
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
    parser.add_argument(
        "env_name", type=str, default="LunarLander-v2", help="Environment name"
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=50000, help="Number of timesteps to train"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    rewards, lengths = lander_learn(
        env_name=args.env_name, num_timesteps=args.num_timesteps, seed=args.seed
    )

    print(f"Final mean reward: {np.mean(rewards[-100:]):.2f}")
if __name__ == "__main__":
    main()