"""
Model-Based Reinforcement Learning Training Script

This script provides modular training for model-based RL algorithms including:
- Q1: Dynamics model training and evaluation
- Q2: MPC with random shooting
- Q3: On-policy MBRL

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import tensorflow.compat.v1 as tf  # type: ignore
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model_based_rl import ModelBasedRLAgent, RandomPolicy
from half_cheetah_env import HalfCheetahEnv
from logger import logger

MUJOCO_ENVS = {
    "HalfCheetah",
    "InvertedPendulum",
    "Hopper",
    "Walker2d",
    "Ant",
    "Humanoid",
}
def create_experiment_dir(exp_name):
    """Create experiment directory and setup logging."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    exp_dir = os.path.join(data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger.setup(exp_name, os.path.join(exp_dir, "log.txt"), "debug")
    return exp_dir
def run_q1(env, args):
    """Run Q1: Train dynamics model and evaluate predictions."""
    print("Running Q1: Dynamics Model Training and Evaluation")
    agent = ModelBasedRLAgent(
        env=env,
        num_init_random_rollouts=args.num_init_random_rollouts,
        max_rollout_length=args.max_rollout_length,
        training_epochs=args.training_epochs,
        training_batch_size=args.training_batch_size,
        mpc_horizon=args.mpc_horizon,
        num_random_action_selection=args.num_random_action_selection,
        nn_layers=args.nn_layers,
    )
    agent.init_tf_sess()
    agent.run_q1()
    print("Evaluating model predictions...")
    random_policy = RandomPolicy(env)
    eval_dataset = agent.gather_rollouts(random_policy, args.num_eval_rollouts)
    data = eval_dataset.get_all()
    for rollout_idx in range(min(args.num_eval_rollouts, 5)):
        states = []
        actions = []
        start_idx = 0
        for i in range(len(data["dones"])):
            if data["dones"][i] or i == len(data["dones"]) - 1:
                rollout_states = data["states"][start_idx : i + 1]
                rollout_actions = data["actions"][start_idx : i + 1]

                if len(rollout_states) > 1:

                    pred_states = [rollout_states[0]]
                    current_state = rollout_states[0]

                    for action in rollout_actions[:-1]:
                        next_state = agent.dynamics_model.predict(
                            current_state.reshape(1, -1),
                            action.reshape(1, -1),
                            agent.sess,
                        )[0]
                        pred_states.append(next_state)
                        current_state = next_state
                    states_array = np.array(rollout_states)
                    pred_states_array = np.array(pred_states)

                    state_dim = states_array.shape[1]
                    rows = int(np.sqrt(state_dim))
                    cols = state_dim // rows

                    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
                    fig.suptitle("Model predictions (red) vs ground truth (black)")

                    for i, (ax, true_state, pred_state) in enumerate(
                        zip(axes.ravel(), states_array.T, pred_states_array.T)
                    ):
                        ax.set_title(f"State {i}")
                        ax.plot(true_state, color="k", label="True")
                        ax.plot(pred_state, color="r", label="Predicted")
                        if i == 0:
                            ax.legend()

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            logger.dir, f"prediction_rollout_{rollout_idx:03d}.jpg"
                        ),
                        bbox_inches="tight",
                    )
                    plt.close()

                start_idx = i + 1
                if rollout_idx >= 4:
                    break

    print(f"All prediction plots saved to {logger.dir}")
def run_q2(env, args):
    """Run Q2: MPC with random shooting."""
    print("Running Q2: MPC with Random Shooting")
    agent = ModelBasedRLAgent(
        env=env,
        num_init_random_rollouts=args.num_init_random_rollouts,
        max_rollout_length=args.max_rollout_length,
        training_epochs=args.training_epochs,
        training_batch_size=args.training_batch_size,
        mpc_horizon=args.mpc_horizon,
        num_random_action_selection=args.num_random_action_selection,
        nn_layers=args.nn_layers,
    )
    agent.init_tf_sess()
    agent.run_q2()
    print("Evaluating MPC policy...")
    eval_dataset = agent.gather_rollouts(
        agent.policy, args.num_eval_rollouts, render=args.render
    )
    returns = []
    rollout_data = eval_dataset.get_all()
    start_idx = 0

    for i in range(len(rollout_data["dones"])):
        if rollout_data["dones"][i] or i == len(rollout_data["dones"]) - 1:
            rewards = rollout_data["rewards"][start_idx : i + 1]
            returns.append(np.sum(rewards))
            start_idx = i + 1

    print(f"MPC Policy Results:")
    print(f"Average Return: {np.mean(returns):.2f}")
    print(f"Std Return: {np.std(returns):.2f}")
    print(f"Min Return: {np.min(returns):.2f}")
    print(f"Max Return: {np.max(returns):.2f}")
def run_q3(env, args):
    """Run Q3: On-policy MBRL."""
    print("Running Q3: On-policy Model-Based RL")
    agent = ModelBasedRLAgent(
        env=env,
        num_init_random_rollouts=args.num_init_random_rollouts,
        max_rollout_length=args.max_rollout_length,
        num_onpolicy_iters=args.num_onpolicy_iters,
        num_onpolicy_rollouts=args.num_onpolicy_rollouts,
        training_epochs=args.training_epochs,
        training_batch_size=args.training_batch_size,
        mpc_horizon=args.mpc_horizon,
        num_random_action_selection=args.num_random_action_selection,
        nn_layers=args.nn_layers,
    )
    agent.init_tf_sess()
    agent.run_q3()
    print("Final evaluation...")
    eval_dataset = agent.gather_rollouts(
        agent.policy, args.num_eval_rollouts, render=args.render
    )
    returns = []
    rollout_data = eval_dataset.get_all()
    start_idx = 0

    for i in range(len(rollout_data["dones"])):
        if rollout_data["dones"][i] or i == len(rollout_data["dones"]) - 1:
            rewards = rollout_data["rewards"][start_idx : i + 1]
            returns.append(np.sum(rewards))
            start_idx = i + 1

    print(f"Final On-policy MBRL Results:")
    print(f"Average Return: {np.mean(returns):.2f}")
    print(f"Std Return: {np.std(returns):.2f}")
    print(f"Min Return: {np.min(returns):.2f}")
    print(f"Max Return: {np.max(returns):.2f}")
def main():
    parser = argparse.ArgumentParser(description="Model-Based Reinforcement Learning")
    parser.add_argument(
        "question",
        type=str,
        choices=["q1", "q2", "q3"],
        help="Question to run (q1, q2, or q3)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetah",
        choices=["HalfCheetah"],
        help="Environment to use",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment during evaluation"
    )
    parser.add_argument(
        "--nn_layers", type=int, default=1, help="Number of layers in dynamics model"
    )
    parser.add_argument(
        "--training_epochs",
        type=int,
        default=60,
        help="Number of training epochs for dynamics model",
    )
    parser.add_argument(
        "--training_batch_size", type=int, default=512, help="Training batch size"
    )
    parser.add_argument(
        "--mpc_horizon", type=int, default=15, help="MPC planning horizon"
    )
    parser.add_argument(
        "--num_random_action_selection",
        type=int,
        default=4096,
        help="Number of random actions for MPC",
    )
    parser.add_argument(
        "--num_init_random_rollouts",
        type=int,
        default=10,
        help="Number of initial random rollouts",
    )
    parser.add_argument(
        "--max_rollout_length", type=int, default=500, help="Maximum rollout length"
    )
    parser.add_argument(
        "--num_eval_rollouts", type=int, default=5, help="Number of evaluation rollouts"
    )
    parser.add_argument(
        "--num_onpolicy_iters",
        type=int,
        default=10,
        help="Number of on-policy iterations",
    )
    parser.add_argument(
        "--num_onpolicy_rollouts",
        type=int,
        default=10,
        help="Number of on-policy rollouts per iteration",
    )

    args = parser.parse_args()
    
    # Check for MuJoCo dependency
    if args.env in MUJOCO_ENVS:
        try:
            import mujoco_py
        except ImportError:
            print("🚫 MuJoCo environment requested but mujoco-py is not installed.")
            print("   Install mujoco-py and GCC 6/7 (brew install gcc --without-multilib) to enable.")
            print("   Skipping run.")
            return
    
    if args.exp_name is None:
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        args.exp_name = f"{args.env}_{args.question}_{timestamp}"
    exp_dir = create_experiment_dir(args.exp_name)
    env = HalfCheetahEnv()
    run_functions = {"q1": run_q1, "q2": run_q2, "q3": run_q3}

    try:
        run_functions[args.question](env, args)
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {exp_dir}")
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        raise
    finally:
        env.close()
if __name__ == "__main__":
    main()