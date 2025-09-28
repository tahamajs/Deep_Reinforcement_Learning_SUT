"""
Multi-Agent Reinforcement Learning Agents

This module implements MADDPG and QMIX agents for multi-agent reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient agent."""

    def __init__(self, obs_dim, action_dim, num_agents, agent_id, lr=1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_id = agent_id

        # Actor network (individual policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)

        # Critic network (centralized, uses global information)
        global_obs_dim = obs_dim * num_agents
        global_action_dim = action_dim * num_agents
        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim + global_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Training parameters
        self.gamma = 0.95
        self.tau = 0.01  # Soft update rate

        # Statistics
        self.actor_losses = []
        self.critic_losses = []

    def get_action(self, observation, exploration_noise=0.1):
        """Get action with optional exploration noise."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            action_probs = self.actor(obs_tensor)

            # Add exploration noise
            if exploration_noise > 0:
                noise = torch.randn_like(action_probs) * exploration_noise
                action_probs = torch.softmax(action_probs + noise, dim=-1)

            action_dist = Categorical(action_probs)
            action = action_dist.sample()

        return action.item()

    def update(self, batch, other_agents):
        """Update MADDPG agent using centralized training."""
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards[:, self.agent_id]).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        batch_size = states.shape[0]

        # Flatten observations for centralized critic
        states_flat = states.view(batch_size, -1)
        next_states_flat = next_states.view(batch_size, -1)

        # Convert actions to one-hot for continuous critic input
        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        actions_flat = actions_onehot.view(batch_size, -1)

        # Get next actions from all agents (target actors)
        next_actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                if i == self.agent_id:
                    next_action_probs = self.actor_target(next_states[:, i])
                else:
                    next_action_probs = other_agents[i].actor_target(next_states[:, i])
                next_actions.append(next_action_probs)

        next_actions_concat = torch.cat(next_actions, dim=-1)

        # Critic update
        with torch.no_grad():
            critic_input = torch.cat([next_states_flat, next_actions_concat], dim=-1)
            target_q_values = self.critic_target(critic_input).squeeze()
            target_q_values = rewards + self.gamma * target_q_values * (~dones)

        current_q_input = torch.cat([states_flat, actions_flat], dim=-1)
        current_q_values = self.critic(current_q_input).squeeze()

        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update
        current_actions = []
        for i in range(self.num_agents):
            if i == self.agent_id:
                current_actions.append(self.actor(states[:, i]))
            else:
                with torch.no_grad():
                    current_actions.append(other_agents[i].actor(states[:, i]))

        current_actions_concat = torch.cat(current_actions, dim=-1)
        actor_critic_input = torch.cat([states_flat, current_actions_concat], dim=-1)
        actor_loss = -self.critic(actor_critic_input).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update()

        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def soft_update(self):
        """Soft update of target networks."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class QMIXAgent:
    """QMIX agent with value function factorization."""

    def __init__(self, obs_dim, action_dim, num_agents, state_dim, lr=1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.state_dim = state_dim

        # Individual Q-networks for each agent
        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ).to(device) for _ in range(num_agents)
        ])

        # Mixing network
        self.mixing_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_agents * 32),  # Weights for mixing
            nn.ReLU()
        ).to(device)

        # Final mixing layer
        self.final_layer = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

        # Target networks
        self.target_q_networks = copy.deepcopy(self.q_networks)
        self.target_mixing_network = copy.deepcopy(self.mixing_network)
        self.target_final_layer = copy.deepcopy(self.final_layer)

        # Optimizers
        all_params = (list(self.q_networks.parameters()) +
                     list(self.mixing_network.parameters()) +
                     list(self.final_layer.parameters()))
        self.optimizer = optim.Adam(all_params, lr=lr)

        # Training parameters
        self.gamma = 0.95
        self.tau = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # Statistics
        self.losses = []
        self.team_rewards = []

    def get_actions(self, observations):
        """Get actions for all agents."""
        actions = []

        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                q_values = self.q_networks[i](obs_tensor)

                # Epsilon-greedy exploration
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.action_dim)
                else:
                    action = q_values.argmax().item()

                actions.append(action)

        return actions

    def mixing_forward(self, individual_q_values, state):
        """Forward pass through mixing network."""
        # Get mixing weights
        mixing_weights = self.mixing_network(state)
        mixing_weights = mixing_weights.view(-1, self.num_agents, 32)

        # Ensure monotonicity (non-negative weights)
        mixing_weights = torch.abs(mixing_weights)

        # Mix individual Q-values
        individual_q_values = individual_q_values.unsqueeze(-1)  # [batch, agents, 1]
        mixed_values = torch.bmm(mixing_weights.transpose(1, 2), individual_q_values)  # [batch, 32, 1]
        mixed_values = mixed_values.squeeze(-1)  # [batch, 32]

        # Final layer
        team_q_value = self.final_layer(mixed_values)

        return team_q_value

    def update(self, batch):
        """Update QMIX agent."""
        states, actions, rewards, next_states, dones = batch

        batch_size = len(states)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        team_rewards = torch.FloatTensor([sum(r) for r in rewards]).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)

        # Flatten states for mixing network
        states_flat = states_tensor.view(batch_size, -1)
        next_states_flat = next_states_tensor.view(batch_size, -1)

        # Get individual Q-values
        individual_q_values = []
        for i in range(self.num_agents):
            q_vals = self.q_networks[i](states_tensor[:, i])
            chosen_q_vals = q_vals.gather(1, actions_tensor[:, i].unsqueeze(1)).squeeze()
            individual_q_values.append(chosen_q_vals)

        individual_q_values = torch.stack(individual_q_values, dim=1)  # [batch, agents]

        # Mix Q-values to get team Q-value
        team_q_values = self.mixing_forward(individual_q_values, states_flat).squeeze()

        # Target Q-values
        with torch.no_grad():
            # Get next individual Q-values
            next_individual_q_values = []
            for i in range(self.num_agents):
                next_q_vals = self.target_q_networks[i](next_states_tensor[:, i])
                max_next_q_vals = next_q_vals.max(1)[0]
                next_individual_q_values.append(max_next_q_vals)

            next_individual_q_values = torch.stack(next_individual_q_values, dim=1)

            # Mix target Q-values
            target_mixing_weights = self.target_mixing_network(next_states_flat)
            target_mixing_weights = target_mixing_weights.view(-1, self.num_agents, 32)
            target_mixing_weights = torch.abs(target_mixing_weights)

            next_individual_q_values_expanded = next_individual_q_values.unsqueeze(-1)
            target_mixed_values = torch.bmm(
                target_mixing_weights.transpose(1, 2),
                next_individual_q_values_expanded
            ).squeeze(-1)

            target_team_q_values = self.target_final_layer(target_mixed_values).squeeze()
            target_team_q_values = team_rewards + self.gamma * target_team_q_values * (~dones_tensor)

        # Compute loss
        loss = F.mse_loss(team_q_values, target_team_q_values)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters()) +
            list(self.mixing_network.parameters()) +
            list(self.final_layer.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        # Update target networks
        if len(self.losses) % 100 == 0:
            self.soft_update_targets()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Store statistics
        self.losses.append(loss.item())
        self.team_rewards.append(team_rewards.mean().item())

        return {
            'loss': loss.item(),
            'team_reward': team_rewards.mean().item(),
            'epsilon': self.epsilon
        }

    def soft_update_targets(self):
        """Soft update of target networks."""
        for target, source in zip(self.target_q_networks, self.q_networks):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_mixing_network.parameters(),
                                      self.mixing_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_final_layer.parameters(),
                                      self.final_layer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)