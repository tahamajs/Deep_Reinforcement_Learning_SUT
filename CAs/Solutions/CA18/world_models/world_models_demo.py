"""
World Models Demo Functions

This module contains demonstration and training functions for world model-based RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from world_models.world_models import WorldModel, MPCPlanner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_world_model_environment():
    """Create a simple continuous control environment for world model training"""

    class ContinuousControlEnv:
        def __init__(self, state_dim=4, action_dim=2):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.max_steps = 200
            self.reset()

        def reset(self):
            self.state = np.random.uniform(-1, 1, self.state_dim)
            self.steps = 0
            return self.state.copy()

        def step(self, action):
            action = np.clip(action, -1, 1)

            next_state = np.zeros_like(self.state)
            next_state[0] = self.state[0] + 0.1 * action[0] + 0.05 * np.sin(self.state[1])
            next_state[1] = self.state[1] + 0.1 * action[1] + 0.02 * self.state[0] * self.state[2]
            next_state[2] = 0.9 * self.state[2] + 0.1 * np.tanh(action[0] + action[1])
            next_state[3] = 0.95 * self.state[3] + 0.1 * np.random.normal(0, 0.1)

            next_state += np.random.normal(0, 0.02, self.state_dim)

            reward = -np.sum(next_state**2) - 0.01 * np.sum(action**2)

            self.steps += 1
            done = self.steps >= self.max_steps or np.linalg.norm(next_state) > 3

            self.state = next_state
            return next_state.copy(), reward, done, {}

    return ContinuousControlEnv()


def collect_random_data(env, n_episodes=100):
    """Collect random interaction data for world model training"""

    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }

    print(f"Collecting {n_episodes} episodes of random data...")

    for episode in range(n_episodes):
        obs = env.reset()
        episode_obs = [obs]
        episode_actions = []
        episode_rewards = []
        episode_dones = []

        while True:
            action = np.random.uniform(-1, 1, env.action_dim)
            next_obs, reward, done, _ = env.step(action)

            episode_obs.append(next_obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)

            if done:
                break

        data['observations'].append(np.array(episode_obs))
        data['actions'].append(np.array(episode_actions))
        data['rewards'].append(np.array(episode_rewards))
        data['dones'].append(np.array(episode_dones))

        if episode % 20 == 0:
            print(f"Episode {episode}/{n_episodes}")

    return data


def create_training_batches(data, batch_size=32, seq_length=20):
    """Create training batches from collected data"""

    batches = []

    for episode_obs, episode_actions, episode_rewards in zip(
        data['observations'], data['actions'], data['rewards']
    ):
        episode_length = len(episode_actions)

        for start_idx in range(0, episode_length - seq_length + 1, seq_length // 2):
            end_idx = start_idx + seq_length

            batch_obs = episode_obs[start_idx:end_idx+1]  # +1 for next obs
            batch_actions = episode_actions[start_idx:end_idx]
            batch_rewards = episode_rewards[start_idx:end_idx]

            batches.append({
                'observations': torch.FloatTensor(batch_obs).to(device),
                'actions': torch.FloatTensor(batch_actions).to(device),
                'rewards': torch.FloatTensor(batch_rewards).unsqueeze(-1).to(device)
            })

    grouped_batches = []
    for i in range(0, len(batches), batch_size):
        batch_group = batches[i:i+batch_size]
        if len(batch_group) == batch_size:

            obs_batch = torch.stack([b['observations'] for b in batch_group])
            action_batch = torch.stack([b['actions'] for b in batch_group])
            reward_batch = torch.stack([b['rewards'] for b in batch_group])

            grouped_batches.append({
                'observations': obs_batch,
                'actions': action_batch,
                'rewards': reward_batch
            })

    return grouped_batches


def train_world_model(world_model, batches, n_epochs=50, lr=1e-3):
    """Train the world model on collected data"""

    optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)

    losses = {'total': [], 'reconstruction': [], 'kl': [], 'reward': []}

    print(f"Training world model for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        epoch_losses = {'total': 0, 'reconstruction': 0, 'kl': 0, 'reward': 0}

        for batch_idx, batch in enumerate(batches):
            obs_seq = batch['observations']  # [batch, seq_len+1, obs_dim]
            action_seq = batch['actions']    # [batch, seq_len, action_dim]
            reward_seq = batch['rewards']    # [batch, seq_len, 1]

            output = world_model.observe_sequence(obs_seq[:, :-1], action_seq)

            recon_loss = F.mse_loss(
                output['reconstructions'],
                obs_seq[:, 1:]  # Target is next observations
            )

            kl_loss = output['kl_losses'].mean() if output['kl_losses'] is not None else 0

            reward_loss = F.mse_loss(output['rewards'], reward_seq)

            total_loss = recon_loss + 0.1 * kl_loss + reward_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['reconstruction'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            epoch_losses['reward'] += reward_loss.item()

        for key in epoch_losses:
            epoch_losses[key] /= len(batches)
            losses[key].append(epoch_losses[key])

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total={epoch_losses['total']:.4f}, "
                  f"Recon={epoch_losses['reconstruction']:.4f}, "
                  f"KL={epoch_losses['kl']:.4f}, "
                  f"Reward={epoch_losses['reward']:.4f}")

    return losses


def evaluate_world_model_planning(env, world_model, planner, n_episodes=10):
    """Evaluate world model with MPC planning"""

    print(f"Evaluating MPC planning for {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0

        state = world_model.rssm.initial_state(1)

        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            embed = world_model.encode(obs_tensor)

            dummy_action = torch.zeros(1, env.action_dim).to(device)
            state = world_model.rssm.observe(embed, dummy_action, state)

            with torch.no_grad():
                action_tensor = planner.plan(state)
                action = action_tensor.cpu().numpy()[0]

            next_obs, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode % 5 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Length={episode_length}")

    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    return episode_rewards