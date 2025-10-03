"""
Video Recording Utilities for Policy Gradient Training

This module provides utilities to record and save videos of agent performance
before and after training.

Author: GitHub Copilot
Date: October 3, 2025
"""

import os
import numpy as np
from gym.wrappers import Monitor
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def record_episode_video(env, agent, video_path, num_episodes=3, max_steps=None):
    """
    Record video of agent performing in environment.
    
    Args:
        env: Gym environment
        agent: Policy gradient agent with sess and policy_net
        video_path: Path to save video (without extension)
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode (None = use env default)
    
    Returns:
        average_return: Average return across recorded episodes
    """
    # Create video directory
    video_dir = os.path.dirname(video_path)
    os.makedirs(video_dir, exist_ok=True)
    
    # Wrap environment with Monitor to record video
    video_name = os.path.basename(video_path)
    env_monitored = Monitor(
        env, 
        video_dir,
        video_callable=lambda episode_id: True,
        force=True,
        name_prefix=video_name
    )
    
    returns = []
    discrete = hasattr(env.action_space, 'n')
    
    try:
        for ep in range(num_episodes):
            ob = env_monitored.reset()
            done = False
            episode_return = 0
            steps = 0
            
            while not done:
                # Get action from policy
                if agent is not None and hasattr(agent, 'sess'):
                    ac = agent.sess.run(
                        agent.policy_net.sy_sampled_ac,
                        feed_dict={agent.policy_net.sy_ob_no: [ob]}
                    )
                    ac = ac[0]
                    if discrete:
                        ac = int(ac)
                else:
                    # Random policy if no agent provided
                    ac = env_monitored.action_space.sample()
                
                ob, reward, done, _ = env_monitored.step(ac)
                episode_return += reward
                steps += 1
                
                if max_steps and steps >= max_steps:
                    break
            
            returns.append(episode_return)
    
    finally:
        env_monitored.close()
    
    avg_return = np.mean(returns) if returns else 0
    return avg_return


def record_before_after_videos(
    env, 
    agent, 
    video_base_path, 
    env_name,
    config_name,
    num_episodes=3
):
    """
    Record videos before training (random policy) and after training.
    
    Args:
        env: Gym environment
        agent: Trained policy gradient agent
        video_base_path: Base directory for videos
        env_name: Name of environment
        config_name: Configuration name (e.g., "Vanilla", "RTG")
        num_episodes: Number of episodes to record
    
    Returns:
        dict with paths and returns for before/after videos
    """
    results = {}
    
    # Create subdirectories
    before_dir = os.path.join(video_base_path, env_name, config_name, "before")
    after_dir = os.path.join(video_base_path, env_name, config_name, "after")
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)
    
    # Record "before" with random policy
    print(f"  📹 Recording BEFORE video (random policy)...")
    before_path = os.path.join(before_dir, "episode")
    before_return = record_episode_video(
        env, 
        None,  # No agent = random policy
        before_path, 
        num_episodes=num_episodes
    )
    results['before'] = {
        'path': before_dir,
        'avg_return': before_return
    }
    print(f"     ✓ Average return (random): {before_return:.2f}")
    
    # Record "after" with trained policy
    print(f"  📹 Recording AFTER video (trained policy)...")
    after_path = os.path.join(after_dir, "episode")
    after_return = record_episode_video(
        env, 
        agent, 
        after_path, 
        num_episodes=num_episodes
    )
    results['after'] = {
        'path': after_dir,
        'avg_return': after_return
    }
    print(f"     ✓ Average return (trained): {after_return:.2f}")
    print(f"     ✓ Improvement: {after_return - before_return:.2f}")
    
    return results


def create_random_agent_wrapper(env):
    """
    Create a dummy object that acts like a random agent.
    Used for "before" videos.
    """
    class RandomAgent:
        def __init__(self, env):
            self.env = env
            self.sess = None
        
        def sample_action(self, obs):
            return self.env.action_space.sample()
    
    return RandomAgent(env)
