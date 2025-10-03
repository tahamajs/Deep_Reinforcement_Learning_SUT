#!/usr/bin/env python
"""
Record demonstration videos for trained policy gradient agents.

This script loads trained agents and records videos of their performance,
comparing before (random) vs after (trained) behavior.

Usage:
    python record_videos.py --logdir results_hw2/logs/CartPole-v0_Vanilla_... --env CartPole-v0

Author: GitHub Copilot
Date: October 3, 2025
"""

import argparse
import os
import sys
import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.policy_gradient import PolicyGradientAgent
from src.video_recorder import record_before_after_videos


def load_trained_agent(logdir, env):
    """
    Load a trained agent from a log directory.
    
    Args:
        logdir: Directory containing params.json and trained model
        env: Gym environment
    
    Returns:
        agent: Loaded PolicyGradientAgent
    """
    # Load parameters
    params_path = os.path.join(logdir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {logdir}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Get environment info
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    # Create agent with same architecture
    agent = PolicyGradientAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        n_layers=params.get('n_layers', 2),
        size=params.get('size', 64),
        learning_rate=params.get('learning_rate', 5e-3),
        gamma=params.get('discount', 1.0),
        reward_to_go=params.get('reward_to_go', False),
        nn_baseline=params.get('nn_baseline', False),
        normalize_advantages=not params.get('dont_normalize_advantages', False),
        min_timesteps_per_batch=params.get('batch_size', 1000),
        max_path_length=env.spec.max_episode_steps,
        animate=False
    )
    
    # Initialize TF session
    agent.init_tf_sess()
    
    # Try to load saved variables
    vars_path = os.path.join(logdir, "vars.pkl")
    if os.path.exists(vars_path):
        import pickle
        with open(vars_path, 'rb') as f:
            saved_vars = pickle.load(f)
        
        # Restore variables
        for var in tf.global_variables():
            if var.name in saved_vars:
                agent.sess.run(var.assign(saved_vars[var.name]))
        print(f"✓ Loaded saved model from {vars_path}")
    else:
        print(f"⚠ No saved model found at {vars_path}, using final trained weights from session")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='Record videos of trained agents')
    parser.add_argument('--logdir', type=str, required=True,
                       help='Path to log directory with trained agent')
    parser.add_argument('--env', type=str, required=True,
                       help='Environment name')
    parser.add_argument('--output_dir', type=str, default='results_hw2/videos',
                       help='Output directory for videos')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes to record')
    
    args = parser.parse_args()
    
    # Create environment
    env = gym.make(args.env)
    
    # Extract config name from logdir
    logdir_name = os.path.basename(args.logdir)
    parts = logdir_name.split('_')
    env_name = parts[0] if len(parts) > 0 else args.env
    config_name = parts[1] if len(parts) > 1 else "Unknown"
    
    print(f"🎬 Recording videos for {env_name} - {config_name}")
    print(f"   Log directory: {args.logdir}")
    
    # Load trained agent
    try:
        agent = load_trained_agent(args.logdir, env)
        print(f"   ✓ Agent loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading agent: {e}")
        print(f"   Will record videos with current agent state")
        return 1
    
    # Record videos
    results = record_before_after_videos(
        env,
        agent,
        args.output_dir,
        env_name,
        config_name,
        num_episodes=args.num_episodes
    )
    
    print(f"\n✅ Videos saved:")
    print(f"   Before: {results['before']['path']}")
    print(f"   After:  {results['after']['path']}")
    
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
