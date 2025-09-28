"""
Offline Reinforcement Learning Algorithms

This module implements Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL)
for offline reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConservativeQNetwork(nn.Module):
    """Q-network for Conservative Q-Learning (CQL)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q-network architecture
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value network for advantage computation
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """Forward pass through Q-network."""
        q_values = self.q_network(state)
        state_value = self.value_network(state)
        return q_values, state_value

    def get_q_values(self, state):
        """Get Q-values for all actions."""
        q_values, _ = self.forward(state)
        return q_values


class ConservativeQLearning:
    """Conservative Q-Learning (CQL) for offline RL."""

    def __init__(self, state_dim, action_dim, lr=3e-4, conservative_weight=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conservative_weight = conservative_weight

        # Networks
        self.q_network = ConservativeQNetwork(state_dim, action_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005  # Target network update rate
        self.update_count = 0

        # Statistics
        self.losses = []
        self.conservative_losses = []
        self.bellman_losses = []

    def compute_conservative_loss(self, states, actions):
        """Compute CQL conservative loss."""
        q_values, _ = self.q_network(states)

        # Log-sum-exp of Q-values (conservative term)
        logsumexp_q = torch.logsumexp(q_values, dim=1)

        # Q-values for behavior policy actions
        behavior_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Conservative loss: penalize high Q-values for unseen actions
        conservative_loss = (logsumexp_q - behavior_q_values).mean()

        return conservative_loss

    def compute_bellman_loss(self, states, actions, rewards, next_states, dones):
        """Compute standard Bellman loss."""
        q_values, _ = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values, _ = self.target_q_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (~dones))

        bellman_loss = F.mse_loss(current_q_values, target_q_values)
        return bellman_loss

    def update(self, batch):
        """Update CQL agent."""
        states, actions, rewards, next_states, dones = batch

        # Compute losses
        conservative_loss = self.compute_conservative_loss(states, actions)
        bellman_loss = self.compute_bellman_loss(states, actions, rewards, next_states, dones)

        # Total loss
        total_loss = self.conservative_weight * conservative_loss + bellman_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.soft_update_target()

        # Store statistics
        self.losses.append(total_loss.item())
        self.conservative_losses.append(conservative_loss.item())
        self.bellman_losses.append(bellman_loss.item())

        return {
            'total_loss': total_loss.item(),
            'conservative_loss': conservative_loss.item(),
            'bellman_loss': bellman_loss.item()
        }

    def soft_update_target(self):
        """Soft update of target network."""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, state, epsilon=0.0):
        """Get action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network.get_q_values(state_tensor)
            return q_values.argmax().item()


class ImplicitQLearning:
    """Implicit Q-Learning (IQL) for offline RL."""

    def __init__(self, state_dim, action_dim, lr=3e-4, expectile=0.7):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expectile = expectile  # Expectile for advantage estimation

        # Networks
        self.q_network = ConservativeQNetwork(state_dim, action_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network).to(device)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005

        # Statistics
        self.q_losses = []
        self.policy_losses = []
        self.advantages = []

    def compute_expectile_loss(self, errors, expectile):
        """Compute expectile loss (asymmetric squared loss)."""
        weights = torch.where(errors > 0, expectile, 1 - expectile)
        return (weights * errors.pow(2)).mean()

    def update_q_function(self, states, actions, rewards, next_states, dones):
        """Update Q-function using expectile regression."""
        q_values, state_values = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            _, next_state_values = self.target_q_network(next_states)
            target_q_values = rewards + (self.gamma * next_state_values.squeeze() * (~dones))

        # Q-function loss using expectile regression
        q_errors = target_q_values - current_q_values
        q_loss = self.compute_expectile_loss(q_errors, 0.5)  # Standard MSE for Q-function

        # Value function loss using expectile for advantage estimation
        advantages = current_q_values.detach() - state_values.squeeze()
        value_loss = self.compute_expectile_loss(advantages, self.expectile)

        total_q_loss = q_loss + value_loss

        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

        return total_q_loss.item(), advantages.mean().item()

    def update_policy(self, states, actions):
        """Update policy using advantage-weighted regression."""
        with torch.no_grad():
            q_values, state_values = self.q_network(states)
            current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            advantages = current_q_values - state_values.squeeze()
            weights = torch.exp(advantages / 3.0).clamp(max=100)  # Temperature scaling

        action_probs = self.policy_network(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)

        policy_loss = -(weights.detach() * log_probs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def update(self, batch):
        """Update IQL agent."""
        states, actions, rewards, next_states, dones = batch

        # Update Q-function and value function
        q_loss, avg_advantage = self.update_q_function(states, actions, rewards, next_states, dones)

        # Update policy
        policy_loss = self.update_policy(states, actions)

        # Soft update target networks
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Store statistics
        self.q_losses.append(q_loss)
        self.policy_losses.append(policy_loss)
        self.advantages.append(avg_advantage)

        return {
            'q_loss': q_loss,
            'policy_loss': policy_loss,
            'avg_advantage': avg_advantage
        }

    def get_action(self, state):
        """Get action from learned policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.policy_network(state_tensor)
            action_dist = Categorical(action_probs)
            return action_dist.sample().item()