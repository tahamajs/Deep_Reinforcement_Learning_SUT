"""
Robust Reinforcement Learning Utilities

This module contains utility functions for robust RL experiments,
including domain randomization, adversarial training helpers, and
robustness evaluation metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_domain_randomized_trajectory(env, agent, max_steps=1000):
    """
    Generate a trajectory in a domain-randomized environment.

    Args:
        env: RobustEnvironment instance
        agent: RL agent
        max_steps: Maximum steps per trajectory

    Returns:
        trajectory: List of (observation, action, reward, log_prob, value, env_params)
    """
    trajectory = []
    observation = env.reset()
    env_params = env.get_current_params()

    for _ in range(max_steps):
        action, log_prob, value = agent.get_action(observation)
        next_observation, reward, done, info = env.step(action)

        trajectory.append((observation, action, reward, log_prob, value, env_params))

        if done:
            break

        observation = next_observation

    return trajectory


def evaluate_robustness(agent, env, num_episodes=10, adversarial_strength=0.1):
    """
    Evaluate agent robustness against adversarial perturbations.

    Args:
        agent: RL agent with adversarial capabilities
        env: Environment to test in
        num_episodes: Number of evaluation episodes
        adversarial_strength: Strength of adversarial perturbations

    Returns:
        metrics: Dictionary of robustness metrics
    """
    original_rewards = []
    adversarial_rewards = []
    perturbation_norms = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _, _ = agent.get_action(obs, use_adversarial=False)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        original_rewards.append(episode_reward)

        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            adv_obs = agent.generate_adversarial_observation(obs)
            perturbation_norms.append(np.linalg.norm(adv_obs - obs))

            action, _, _ = agent.get_action(obs, use_adversarial=False)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        adversarial_rewards.append(episode_reward)

    return {
        "original_mean_reward": np.mean(original_rewards),
        "original_std_reward": np.std(original_rewards),
        "adversarial_mean_reward": np.mean(adversarial_rewards),
        "adversarial_std_reward": np.std(adversarial_rewards),
        "robustness_drop": np.mean(original_rewards) - np.mean(adversarial_rewards),
        "mean_perturbation_norm": np.mean(perturbation_norms),
        "std_perturbation_norm": np.std(perturbation_norms),
    }


def compute_policy_divergence(policy1, policy2, observations):
    """
    Compute KL divergence between two policies over a set of observations.

    Args:
        policy1: First policy network
        policy2: Second policy network
        observations: Batch of observations

    Returns:
        mean_kl: Mean KL divergence
    """
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(observations).to(device)

        probs1 = policy1(obs_tensor)
        probs2 = policy2(obs_tensor)

        kl_div = (
            F.kl_div(torch.log(probs2 + 1e-8), probs1, reduction="none")
            .sum(dim=-1)
            .mean()
        )

    return kl_div.item()


def create_ensemble_policies(base_agent, num_policies=5, noise_std=0.1):
    """
    Create an ensemble of policies by adding noise to the base policy.

    Args:
        base_agent: Base agent with policy network
        num_policies: Number of policies in ensemble
        noise_std: Standard deviation of noise to add

    Returns:
        ensemble: List of policy networks
    """
    ensemble = []

    for _ in range(num_policies):
        policy_copy = copy.deepcopy(base_agent.policy_network)

        with torch.no_grad():
            for param in policy_copy.parameters():
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)

        ensemble.append(policy_copy)

    return ensemble


def evaluate_ensemble_uncertainty(ensemble, observation):
    """
    Evaluate uncertainty in ensemble predictions.

    Args:
        ensemble: List of policy networks
        observation: Single observation

    Returns:
        uncertainty_metrics: Dictionary with uncertainty measures
    """
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)

        predictions = []
        for policy in ensemble:
            probs = policy(obs_tensor).squeeze().cpu().numpy()
            predictions.append(probs)

        predictions = np.array(predictions)

        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        entropy = -np.sum(mean_prediction * np.log(mean_prediction + 1e-8))

        action_variances = np.var(predictions, axis=0)

    return {
        "mean_entropy": entropy,
        "max_action_variance": np.max(action_variances),
        "mean_action_variance": np.mean(action_variances),
        "prediction_std": np.mean(std_prediction),
    }


def generate_diverse_environments(base_env_class, num_variations=10):
    """
    Generate diverse environment variations for domain randomization.

    Args:
        base_env_class: Base environment class
        num_variations: Number of variations to generate

    Returns:
        environments: List of environment instances with different parameters
    """
    environments = []

    for i in range(num_variations):
        size_variation = np.random.uniform(0.8, 1.2)
        noise_variation = np.random.uniform(0.0, 0.2)
        reward_scale = np.random.uniform(0.8, 1.2)

        env = base_env_class(
            size=int(10 * size_variation),
            noise_level=noise_variation,
            reward_scale=reward_scale,
        )

        environments.append(env)

    return environments


def collect_robust_training_data(environments, agent, trajectories_per_env=5):
    """
    Collect training data across diverse environments.

    Args:
        environments: List of environment instances
        agent: RL agent
        trajectories_per_env: Number of trajectories per environment

    Returns:
        all_trajectories: Combined trajectories from all environments
    """
    all_trajectories = []

    for env in environments:
        for _ in range(trajectories_per_env):
            trajectory = generate_domain_randomized_trajectory(env, agent)
            all_trajectories.append(trajectory)

    return all_trajectories


def compute_robustness_metrics(trajectories):
    """
    Compute robustness metrics from collected trajectories.

    Args:
        trajectories: List of trajectories with environment parameters

    Returns:
        metrics: Dictionary of robustness metrics
    """
    if not trajectories:
        return {}

    env_sizes = []
    noise_levels = []
    rewards = []

    for trajectory in trajectories:
        for step in trajectory:
            obs, action, reward, log_prob, value, env_params = step
            env_sizes.append(env_params["environment_size"])
            noise_levels.append(env_params["noise_level"])
            rewards.append(reward)

    return {
        "environment_size_diversity": len(set(env_sizes)),
        "noise_level_range": np.ptp(noise_levels),
        "mean_reward": np.mean(rewards),
        "reward_std": np.std(rewards),
        "total_trajectories": len(trajectories),
        "total_steps": sum(len(traj) for traj in trajectories),
    }
