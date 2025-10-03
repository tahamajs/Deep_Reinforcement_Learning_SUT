"""
Actor-Critic Training Script

This script trains an Actor-Critic agent on gym environments.

Usage:
    python run_ac.py [options]

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
import logz
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.actor_critic import ActorCriticAgent

MUJOCO_ENVS = {
    "InvertedPendulum-v1",
    "InvertedPendulum-v2",
    "HalfCheetah-v1",
    "HalfCheetah-v2",
    "Hopper-v1",
    "Hopper-v2",
    "Walker2d-v1",
    "Walker2d-v2",
    "Ant-v1",
    "Ant-v2",
    "Humanoid-v1",
    "Humanoid-v2",
}
def train_AC(
    exp_name,
    env_name,
    n_iter,
    gamma,
    min_timesteps_per_batch,
    max_path_length,
    learning_rate,
    num_target_updates,
    num_grad_steps_per_target_update,
    animate,
    logdir,
    normalize_advantages,
    seed,
    n_layers,
    size,
):
    """Train Actor-Critic agent."""
    start = time.time()
    logz.configure_output_dir(logdir)
    args = {
        "exp_name": exp_name,
        "env_name": env_name,
        "n_iter": n_iter,
        "gamma": gamma,
        "min_timesteps_per_batch": min_timesteps_per_batch,
        "max_path_length": max_path_length,
        "learning_rate": learning_rate,
        "num_target_updates": num_target_updates,
        "num_grad_steps_per_target_update": num_grad_steps_per_target_update,
        "animate": animate,
        "logdir": logdir,
        "normalize_advantages": normalize_advantages,
        "seed": seed,
        "n_layers": n_layers,
        "size": size,
    }
    logz.save_params(args)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    if env_name in MUJOCO_ENVS:
        raise RuntimeError(
            f"Environment '{env_name}' requires MuJoCo (mujoco-py) and a GCC 6/7 toolchain. "
            "Install dependencies (e.g., brew install gcc --without-multilib) before running."
        )

    env = gym.make(env_name)
    env.seed(seed)
    max_path_length = max_path_length or env.spec.max_episode_steps
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    computation_graph_args = {
        "n_layers": n_layers,
        "ob_dim": ob_dim,
        "ac_dim": ac_dim,
        "discrete": discrete,
        "size": size,
        "learning_rate": learning_rate,
        "num_target_updates": num_target_updates,
        "num_grad_steps_per_target_update": num_grad_steps_per_target_update,
    }

    sample_trajectory_args = {
        "animate": animate,
        "max_path_length": max_path_length,
        "min_timesteps_per_batch": min_timesteps_per_batch,
    }

    estimate_advantage_args = {
        "gamma": gamma,
        "normalize_advantages": normalize_advantages,
    }

    agent = ActorCriticAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        n_layers=n_layers,
        size=size,
        learning_rate=learning_rate,
        num_target_updates=num_target_updates,
        num_grad_steps_per_target_update=num_grad_steps_per_target_update,
        gamma=gamma,
        normalize_advantages=normalize_advantages,
        max_path_length=max_path_length,
        min_timesteps_per_batch=min_timesteps_per_batch,
        animate=animate,
    )
    agent.build_computation_graph()
    agent.init_tf_sess()
    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])
        agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
        adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        agent.update_actor(ob_no, ac_na, adv_n)
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [len(path["reward"]) for path in paths]
        logz.log_tabular("Time", time.time() - start)
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
        logz.pickle_tf_vars(agent.sess)
def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--exp_name", type=str, default="vac")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--n_iter", "-n", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=1000)
    parser.add_argument("--ep_len", "-ep", type=float, default=-1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--dont_normalize_advantages", "-dna", action="store_true")
    parser.add_argument("--num_target_updates", "-ntu", type=int, default=10)
    parser.add_argument(
        "--num_grad_steps_per_target_update", "-ngsptu", type=int, default=10
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_experiments", "-e", type=int, default=1)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=64)
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10 * e

        if args.env_name in MUJOCO_ENVS:
            print(
                "🚫 MuJoCo environment requested. Install mujoco-py and GCC 6/7 (brew install gcc --without-multilib) to enable. Skipping run."
            )
            continue

        print("Running experiment with seed %d" % seed)

        logdir = (
            args.exp_name
            + "_"
            + args.env_name
            + "_"
            + time.strftime("%d-%m-%Y_%H-%M-%S")
        )
        logdir = os.path.join(data_path, logdir)

        # Call train_AC directly instead of using multiprocessing
        train_AC(
            exp_name=args.exp_name,
            env_name=args.env_name,
            n_iter=args.n_iter,
            gamma=args.discount,
            min_timesteps_per_batch=args.batch_size,
            max_path_length=max_path_length,
            learning_rate=args.learning_rate,
            num_target_updates=args.num_target_updates,
            num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
            animate=args.render,
            logdir=logdir,
            normalize_advantages=not (args.dont_normalize_advantages),
            seed=seed,
            n_layers=args.n_layers,
            size=args.size,
        )

if __name__ == "__main__":
    main()