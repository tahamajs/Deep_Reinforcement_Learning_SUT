#!/usr/bin/env python

"""
Modular Behavioral Cloning Implementation

This script provides a modular implementation of behavioral cloning for imitation learning.

Usage:
    # Collect expert data
    python run_bc.py --mode collect --expert_policy experts/Humanoid-v2.pkl --env Humanoid-v2 --num_rollouts 20

    # Train BC policy
    python run_bc.py --mode train --data_file expert_data/Humanoid-v2.pkl --epochs 100

    # Evaluate BC policy
    python run_bc.py --mode evaluate --model_file models/bc_policy.pkl --episodes 10

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import os
import argparse
import pickle
import gym
from src.expert_data_collector import ExpertDataCollector, load_expert_policy
from src.behavioral_cloning import BehavioralCloning


def collect_expert_data(args):
    """Collect expert demonstration data."""
    print('Loading expert policy...')
    policy_fn = load_expert_policy(args.expert_policy)

    print('Creating data collector...')
    collector = ExpertDataCollector(policy_fn, args.env, args.max_timesteps)

    print(f'Collecting {args.num_rollouts} expert rollouts...')
    data = collector.collect_rollouts(args.num_rollouts, args.render)

    # Save data
    os.makedirs('expert_data', exist_ok=True)
    data_file = os.path.join('expert_data', f'{args.env}.pkl')
    collector.save_data(data, data_file)

    print(f'Expert data collection completed!')
    print(f'Mean return: {data["mean_return"]:.2f}')
    print(f'Std return: {data["std_return"]:.2f}')


def train_bc_policy(args):
    """Train behavioral cloning policy."""
    # Load expert data
    print(f'Loading expert data from {args.data_file}...')
    with open(args.data_file, 'rb') as f:
        expert_data = pickle.load(f)

    observations = expert_data['observations']
    actions = expert_data['actions']

    print(f'Loaded {len(observations)} expert transitions')

    # Create environment to get dimensions
    env = gym.make(args.env)

    # Create and train BC agent
    print('Training behavioral cloning policy...')
    bc_agent = BehavioralCloning(env, args.learning_rate, args.hidden_sizes)

    history = bc_agent.train(observations, actions, args.epochs, args.batch_size)

    # Save trained model
    os.makedirs('models', exist_ok=True)
    model_file = os.path.join('models', 'bc_policy.pkl')

    # Save model weights
    saver = tf.train.Saver()
    saver.save(bc_agent.sess, model_file)
    print(f'Model saved to {model_file}')

    # Save training history
    history_file = os.path.join('models', 'training_history.pkl')
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)

    print('Training completed!')


def evaluate_bc_policy(args):
    """Evaluate trained behavioral cloning policy."""
    # Create environment
    env = gym.make(args.env)

    # Create BC agent
    bc_agent = BehavioralCloning(env)

    # Load trained model
    print(f'Loading model from {args.model_file}...')
    saver = tf.train.Saver()
    saver.restore(bc_agent.sess, args.model_file)

    print(f'Evaluating policy for {args.episodes} episodes...')
    results = bc_agent.evaluate(args.episodes, args.render)

    print('Evaluation Results:')
    print(f'Mean Return: {results["mean_return"]:.2f}')
    print(f'Std Return: {results["std_return"]:.2f}')


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning')

    # Common arguments
    parser.add_argument('--env', type=str, required=True,
                       help='Gym environment name')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')

    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Collect mode
    collect_parser = subparsers.add_parser('collect', help='Collect expert data')
    collect_parser.add_argument('--expert_policy', type=str, required=True,
                               help='Path to expert policy file')
    collect_parser.add_argument('--num_rollouts', type=int, default=20,
                               help='Number of expert rollouts to collect')
    collect_parser.add_argument('--max_timesteps', type=int,
                               help='Maximum timesteps per episode')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train BC policy')
    train_parser.add_argument('--data_file', type=str, required=True,
                             help='Path to expert data file')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=64,
                             help='Training batch size')
    train_parser.add_argument('--learning_rate', type=float, default=1e-3,
                             help='Learning rate')
    train_parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[100, 100],
                             help='Hidden layer sizes')

    # Evaluate mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate BC policy')
    eval_parser.add_argument('--model_file', type=str, required=True,
                            help='Path to trained model file')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')

    args = parser.parse_args()

    if args.mode == 'collect':
        collect_expert_data(args)
    elif args.mode == 'train':
        train_bc_policy(args)
    elif args.mode == 'evaluate':
        evaluate_bc_policy(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    import tensorflow as tf
    main()