"""
Data Collection Utilities
"""

import torch
import random
from typing import List, Dict, Any


def collect_world_model_data(env, n_episodes=100):
    """Collect data for world model training"""
    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': []
    }

    for episode in range(n_episodes):
        obs = env.reset()

        for step in range(200):  # Max episode length
            action = env.sample_action()
            next_obs, reward, done = env.step(action)

            data['observations'].append(obs)
            data['actions'].append([action])
            data['rewards'].append(reward)
            data['next_observations'].append(next_obs)

            obs = next_obs
            if done:
                break

    # Convert to tensors
    for key in data:
        data[key] = torch.FloatTensor(data[key])

    return data


def collect_sequence_data(env, n_episodes=100, seq_length=20):
    """Collect sequential data for RSSM training"""
    sequences = []

    for episode in range(n_episodes):
        obs_sequence = []
        action_sequence = []
        reward_sequence = []

        obs = env.reset()

        for t in range(seq_length):
            # Random policy
            action = np.random.randint(0, 2)
            next_obs, reward, done = env.step(action)

            obs_sequence.append(obs)
            action_sequence.append([action])  # Make it 1D
            reward_sequence.append(reward)

            obs = next_obs

            if done:
                break

        if len(obs_sequence) >= seq_length:
            sequences.append({
                'observations': obs_sequence[:seq_length],
                'actions': action_sequence[:seq_length],
                'rewards': reward_sequence[:seq_length]
            })

    return sequences


def prepare_rssm_batch(sequences, batch_size=32):
    """Prepare batch for RSSM training"""
    # Randomly sample sequences
    batch_sequences = random.sample(sequences, min(batch_size, len(sequences)))

    observations = []
    actions = []
    rewards = []

    for seq in batch_sequences:
        observations.append(seq['observations'])
        actions.append(seq['actions'])
        rewards.append(seq['rewards'])

    # Convert to tensors
    observations = torch.FloatTensor(observations)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)

    return observations, actions, rewards