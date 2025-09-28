#!/usr/bin/env python3
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
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from sac_agent import SACAgent
from exploration_agent import ExplorationAgent, DiscreteExplorationAgent
from meta_agent import MetaLearningAgent, MAMLAgent


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

    # Create SAC agent
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

    # Initialize TensorFlow
    agent.init_tf_sess()

    # Training loop
    episode_rewards = []
    episode_lengths = []

    state = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(args.total_steps):
        # Get action
        action = agent.get_action(state)

        # Take step
        next_state, reward, done, _ = env.step(action)

        # Add experience
        agent.add_experience(state, action, reward, next_state, done)

        # Train
        agent.train_step()

        # Update counters
        episode_reward += reward
        episode_length += 1
        state = next_state

        # Handle episode end
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

    # Save results
    np.save(os.path.join(args.exp_dir, "episode_rewards.npy"), episode_rewards)
    np.save(os.path.join(args.exp_dir, "episode_lengths.npy"), episode_lengths)

    print(f"SAC training completed. Results saved to {args.exp_dir}")


def run_exploration(env, args):
    """Run exploration training."""
    print("Running exploration training...")

    # Create exploration agent
    if hasattr(env.action_space, "n"):  # Discrete action space
        agent = DiscreteExplorationAgent(
            state_dim=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            bonus_coeff=args.bonus_coeff,
        )
    else:  # Continuous action space
        agent = ExplorationAgent(
            state_dim=env.observation_space.shape[0], bonus_coeff=args.bonus_coeff
        )

    # Initialize TensorFlow
    agent.init_tf_sess()

    # Collect initial data
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

    # Fit density model
    print("Fitting density model...")
    if hasattr(agent, "fit_density_model"):
        if hasattr(env.action_space, "n"):
            # For discrete actions, we need action data too
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

    # Test exploration bonus
    print("Testing exploration bonus...")
    test_states = states[:10]  # Test on first 10 states

    if hasattr(env.action_space, "n"):
        test_actions = np.zeros((10, env.action_space.n))
        test_actions[:, 0] = 1  # First action
        bonuses = agent.compute_reward_bonus(test_states, test_actions)
    else:
        bonuses = agent.compute_reward_bonus(test_states)

    print(f"Sample bonuses: {bonuses}")

    # Test reward modification
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

    # Create meta-learning agent
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

    # Initialize TensorFlow
    agent.init_tf_sess()

    # Generate synthetic tasks (simplified - in practice would use different environments)
    print("Generating task data...")

    for task_id in range(args.num_tasks):
        # Collect trajectory for this task
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

    # Meta-training
    print("Starting meta-training...")
    for meta_step in range(args.meta_steps):
        agent.meta_train_step()

        if meta_step % args.log_interval == 0:
            print(f"Meta step {meta_step}/{args.meta_steps}")

    print(f"Meta-learning training completed. Results saved to {args.exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Homework 5 Training")

    # Algorithm selection
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["sac", "exploration", "meta"],
        help="Algorithm to run",
    )

    # Experiment settings
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--env_name", type=str, default="Pendulum-v0", help="Environment name"
    )

    # SAC parameters
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

    # Exploration parameters
    parser.add_argument(
        "--bonus_coeff", type=float, default=1.0, help="Exploration bonus coefficient"
    )
    parser.add_argument(
        "--initial_rollouts",
        type=int,
        default=10,
        help="Initial rollouts for density estimation",
    )

    # Meta-learning parameters
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

    # General parameters
    parser.add_argument(
        "--total_steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument(
        "--max_episode_length", type=int, default=1000, help="Maximum episode length"
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")

    args = parser.parse_args()

    # Create experiment name
    if args.exp_name is None:
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        args.exp_name = f"{args.algorithm}_{args.env_name}_{timestamp}"

    # Create experiment directory
    args.exp_dir = create_experiment_dir(args.exp_name)

    # Import and create environment
    try:
        import gym

        env = gym.make(args.env_name)
    except ImportError:
        print("Gym not installed. Please install with: pip install gym")
        return

    # Run selected algorithm
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
