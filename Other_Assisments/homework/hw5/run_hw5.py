"""
Homework 5 Training Script

This script provides modular training for:
- SAC (Soft Actor-Critic)
- Exploration methods
- Meta-learning

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
try:
    import tensorflow.compat.v1 as tf  # type: ignore
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agents.sac_agent import SACAgent
from agents.exploration_agent import ExplorationAgent, DiscreteExplorationAgent
from agents.meta_agent import MetaLearningAgent, MAMLAgent

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
    "SparseHalfCheetah-v0",
}
def create_experiment_dir(exp_name):
    """Create experiment directory."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    exp_dir = os.path.join(data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
def run_sac(env, args):
    """Run SAC training."""
    print("Running SAC training...")
    agent = SACAgent(
        env=env,
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        batch_size=args.batch_size,
        discount=args.discount,
        tau=args.tau,
        reparameterize=args.reparameterize,
    )
    agent.init_tf_sess()
    episode_rewards = []
    episode_lengths = []

    state = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(args.total_steps):

        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.add_experience(state, action, reward, next_state, done)
        agent.train_step()
        episode_reward += reward
        episode_length += 1
        state = next_state
        if done or episode_length >= args.max_episode_length:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if len(episode_rewards) % args.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-args.log_interval :])
                avg_length = np.mean(episode_lengths[-args.log_interval :])
                print(
                    f"Episode {len(episode_rewards)}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}"
                )

            state = env.reset()
            episode_reward = 0
            episode_length = 0
    np.save(os.path.join(args.exp_dir, "episode_rewards.npy"), episode_rewards)
    np.save(os.path.join(args.exp_dir, "episode_lengths.npy"), episode_lengths)

    print(f"SAC training completed. Results saved to {args.exp_dir}")
def run_exploration(env, args):
    """Run exploration training."""
    print("Running exploration training...")
    if hasattr(env.action_space, "n"):
        agent = DiscreteExplorationAgent(
            state_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            bonus_coeff=args.bonus_coeff,
        )
    else:
        agent = ExplorationAgent(
            state_dim=env.observation_space.shape[0], bonus_coeff=args.bonus_coeff
        )
    agent.init_tf_sess()
    print("Collecting initial exploration data...")
    states = []

    for _ in range(args.initial_rollouts):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < args.max_episode_length:
            if hasattr(env.action_space, "n"):
                action = env.action_space.sample()
                action_onehot = np.zeros(env.action_space.n)
                action_onehot[action] = 1
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            states.append(state)

            state = next_state
            steps += 1

    states = np.array(states)
    print("Fitting density model...")
    if hasattr(agent, "fit_density_model"):
        if hasattr(env.action_space, "n"):

            actions = []
            for _ in range(args.initial_rollouts):
                state = env.reset()
                done = False
                steps = 0

                while not done and steps < args.max_episode_length:
                    action = env.action_space.sample()
                    action_onehot = np.zeros(env.action_space.n)
                    action_onehot[action] = 1

                    next_state, reward, done, _ = env.step(action)
                    actions.append(action_onehot)

                    state = next_state
                    steps += 1

            actions = np.array(actions)
            agent.fit_density_model(states, actions)
        else:
            agent.fit_density_model(states)
    print("Testing exploration bonus...")
    test_states = states[:10]

    if hasattr(env.action_space, "n"):
        test_actions = np.zeros((10, env.action_space.n))
        test_actions[:, 0] = 1
        bonuses = agent.compute_reward_bonus(test_states, test_actions)
    else:
        bonuses = agent.compute_reward_bonus(test_states)

    print(f"Sample bonuses: {bonuses}")
    test_rewards = np.ones(10)
    if hasattr(env.action_space, "n"):
        modified_rewards = agent.modify_reward(test_rewards, test_states, test_actions)
    else:
        modified_rewards = agent.modify_reward(test_rewards, test_states)

    print(f"Original rewards: {test_rewards}")
    print(f"Modified rewards: {modified_rewards}")

    print(f"Exploration training completed. Results saved to {args.exp_dir}")
def run_meta_learning(env, args):
    """Run meta-learning training."""
    print("Running meta-learning training...")
    if args.algorithm == "maml":
        agent = MAMLAgent(
            env=env,
            hidden_sizes=args.hidden_sizes,
            learning_rate=args.learning_rate,
            meta_learning_rate=args.meta_learning_rate,
            adaptation_steps=args.adaptation_steps,
            meta_batch_size=args.meta_batch_size,
            discount=args.discount,
        )
    else:
        agent = MetaLearningAgent(
            env=env,
            hidden_sizes=args.hidden_sizes,
            learning_rate=args.learning_rate,
            meta_learning_rate=args.meta_learning_rate,
            adaptation_steps=args.adaptation_steps,
            meta_batch_size=args.meta_batch_size,
            discount=args.discount,
        )
    agent.init_tf_sess()
    print("Generating task data...")

    for task_id in range(args.num_tasks):

        states = []
        actions = []
        rewards = []
        dones = []

        state = env.reset()
        done = False
        steps = 0

        while not done and steps < args.max_episode_length:
            action, _ = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            steps += 1

        trajectory = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
        }

        agent.add_task_trajectory(trajectory)
    print("Starting meta-training...")
    for meta_step in range(args.meta_steps):
        agent.meta_train_step()

        if meta_step % args.log_interval == 0:
            print(f"Meta step {meta_step}/{args.meta_steps}")

    print(f"Meta-learning training completed. Results saved to {args.exp_dir}")
def main():
    parser = argparse.ArgumentParser(description="Homework 5 Training")
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["sac", "exploration", "meta"],
        help="Algorithm to run",
    )
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--env_name", type=str, default="Pendulum-v0", help="Environment name"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-3, help="Learning rate"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="SAC temperature parameter"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--tau", type=float, default=0.01, help="Soft update coefficient"
    )
    parser.add_argument(
        "--reparameterize", action="store_true", help="Use reparameterization trick"
    )
    parser.add_argument(
        "--bonus_coeff", type=float, default=1.0, help="Exploration bonus coefficient"
    )
    parser.add_argument(
        "--initial_rollouts",
        type=int,
        default=10,
        help="Initial rollouts for density estimation",
    )
    parser.add_argument(
        "--meta_learning_rate", type=float, default=1e-3, help="Meta learning rate"
    )
    parser.add_argument(
        "--adaptation_steps", type=int, default=5, help="Adaptation steps"
    )
    parser.add_argument(
        "--meta_batch_size", type=int, default=4, help="Meta batch size"
    )
    parser.add_argument(
        "--num_tasks", type=int, default=20, help="Number of tasks for meta-training"
    )
    parser.add_argument(
        "--meta_steps", type=int, default=100, help="Number of meta-training steps"
    )
    parser.add_argument(
        "--total_steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument(
        "--max_episode_length", type=int, default=1000, help="Maximum episode length"
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")

    args = parser.parse_args()
    
    # Check for MuJoCo dependency
    if args.env_name in MUJOCO_ENVS:
        try:
            import mujoco_py
        except ImportError:
            print("🚫 MuJoCo environment requested but mujoco-py is not installed.")
            print("   Install mujoco-py and GCC 6/7 (brew install gcc --without-multilib) to enable.")
            print("   Skipping run.")
            return
    
    if args.exp_name is None:
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        args.exp_name = f"{args.algorithm}_{args.env_name}_{timestamp}"
    
    args.exp_dir = create_experiment_dir(args.exp_name)
    try:
        import gym

        env = gym.make(args.env_name)
    except ImportError:
        print("Gym not installed. Please install with: pip install gym")
        return
    try:
        if args.algorithm == "sac":
            run_sac(env, args)
        elif args.algorithm == "exploration":
            run_exploration(env, args)
        elif args.algorithm == "meta":
            run_meta_learning(env, args)

        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {args.exp_dir}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        env.close()
if __name__ == "__main__":
    main()