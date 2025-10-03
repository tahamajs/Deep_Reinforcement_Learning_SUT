"""
Policy Gradient Training Script

This script trains a policy gradient agent on gym environments.

Usage:
    python run_pg.py [options]

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
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.policy_gradient import PolicyGradientAgent, pathlength
from src.utils import flatten_list_of_rollouts
from src.video_recorder import record_before_after_videos
import logz
def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--exp_name", type=str, default="vpg")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--n_iter", "-n", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=1000)
    parser.add_argument("--ep_len", "-ep", type=float, default=-1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--dont_normalize_advantages", "-dna", action="store_true")
    parser.add_argument("--nn_baseline", "-bl", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_experiments", "-e", type=int, default=1)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=64)
    parser.add_argument("--record_video", action="store_true",
                       help="Record before/after training videos")

    args = parser.parse_args()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env_name)
    max_path_length = args.ep_len if args.ep_len > 0 else env.spec.max_episode_steps
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    for e in range(args.n_experiments):

        logdir = (
            args.exp_name
            + "_"
            + args.env_name
            + "_"
            + time.strftime("%d-%m-%Y_%H-%M-%S")
        )
        logz.configure_output_dir(logdir)
        logz.save_params(vars(args))
        agent = PolicyGradientAgent(
            ob_dim=ob_dim,
            ac_dim=ac_dim,
            discrete=discrete,
            n_layers=args.n_layers,
            size=args.size,
            learning_rate=args.learning_rate,
            gamma=args.discount,
            reward_to_go=args.reward_to_go,
            nn_baseline=args.nn_baseline,
            normalize_advantages=not (args.dont_normalize_advantages),
            min_timesteps_per_batch=args.batch_size,
            max_path_length=max_path_length,
            animate=args.render,
        )
        agent.init_tf_sess()
        
        # Record "before" video (iteration 0, untrained)
        if args.record_video:
            print("📹 Recording BEFORE training video...")
            try:
                env_temp = gym.make(args.env_name)
                before_results = record_before_after_videos(
                    env_temp,
                    agent,  # Untrained agent
                    "results_hw2/videos",
                    args.env_name,
                    args.exp_name,
                    num_episodes=1
                )
                env_temp.close()
                print("✓ BEFORE video recorded")
            except Exception as e:
                print(f"⚠ Could not record BEFORE video: {e}")
        
        total_timesteps = 0
        for itr in range(args.n_iter):
            print("********** Iteration %i ************" % itr)
            paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
            total_timesteps += timesteps_this_batch
            ob_no, ac_na, re_n = flatten_list_of_rollouts(paths)
            q_n, adv_n = agent.estimate_return(
                ob_no, [path["reward"] for path in paths]
            )
            agent.update_parameters(ob_no, ac_na, q_n, adv_n)
            returns = [path["reward"].sum() for path in paths]
            ep_lengths = [pathlength(path) for path in paths]

            logz.log_tabular("Time", time.time())
            logz.log_tabular("Iteration", itr)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("EpLenMean", np.mean(ep_lengths))
            logz.log_tabular("EpLenStd", np.std(ep_lengths))
            logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            logz.dump_tabular()

        # Save trained model variables
        logz.pickle_tf_vars(agent.sess)

        # Record "after" video (trained policy)
        if args.record_video:
            print("\n📹 Recording AFTER training video...")
            try:
                env_temp = gym.make(args.env_name)
                after_results = record_before_after_videos(
                    env_temp,
                    agent,  # Trained agent
                    "results_hw2/videos",
                    args.env_name,
                    args.exp_name,
                    num_episodes=3
                )
                env_temp.close()
                print("✓ AFTER video recorded")
                print(f"   Improvement: {after_results['after']['avg_return'] - after_results['before']['avg_return']:.2f}")
            except Exception as e:
                print(f"⚠ Could not record AFTER video: {e}")

        print("Training completed!")
if __name__ == "__main__":
    main()