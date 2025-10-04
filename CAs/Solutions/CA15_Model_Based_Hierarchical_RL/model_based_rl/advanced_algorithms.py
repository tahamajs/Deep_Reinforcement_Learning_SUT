"""
Advanced Model-Based RL Algorithms

This module contains advanced model-based RL implementations including:
- Probabilistic dynamics models with uncertainty quantification
- Model-based policy optimization (MBPO)
- Dreamer and DreamerV2 algorithms
- Model-based meta-learning
- Safe model-based RL with constraints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, MultivariateNormal
import random
import copy
from collections import deque
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbabilisticDynamicsModel(nn.Module):
    """Probabilistic dynamics model with epistemic and aleatoric uncertainty."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, ensemble_size=5):
        super(ProbabilisticDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Ensemble of networks for epistemic uncertainty
        self.ensemble = nn.ModuleList()
        for _ in range(ensemble_size):
            net = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim * 2 + 1),  # mean, log_std, reward
            )
            self.ensemble.append(net)

    def forward(self, state, action):
        """Forward pass through ensemble."""
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([state, action], dim=-1)

        predictions = []
        for net in self.ensemble:
            output = net(input_tensor)
            next_state_mean = output[:, : self.state_dim]
            next_state_log_std = output[:, self.state_dim : self.state_dim * 2]
            reward = output[:, self.state_dim * 2 :]
            predictions.append((next_state_mean, next_state_log_std, reward))

        return predictions

    def sample_prediction(self, state, action):
        """Sample from ensemble predictions."""
        predictions = self.forward(state, action)

        # Randomly select one ensemble member
        idx = random.randint(0, self.ensemble_size - 1)
        mean, log_std, reward = predictions[idx]

        # Sample from normal distribution
        std = torch.exp(log_std)
        next_state = torch.normal(mean, std)

        return next_state, reward.squeeze()


class ModelBasedPolicyOptimization(nn.Module):
    """Model-Based Policy Optimization (MBPO) implementation."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ModelBasedPolicyOptimization, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean and log_std
        )

        # Critic networks
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.tau = 0.005
        self.gamma = 0.99
        self.alpha = 0.2  # Temperature parameter

    def get_action(self, state, deterministic=False):
        """Get action from policy."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        output = self.actor(state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))

        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        return action.clamp(-1, 1)

    def update(self, states, actions, rewards, next_states, dones):
        """Update actor and critic networks."""
        # Update critics
        with torch.no_grad():
            next_actions = self.get_action(next_states)
            next_q1 = self.target_critic1(
                torch.cat([next_states, next_actions], dim=-1)
            )
            next_q2 = self.target_critic2(
                torch.cat([next_states, next_actions], dim=-1)
            )
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_actions = self.get_action(states)
        q1_new = self.critic1(torch.cat([states, new_actions], dim=-1))
        q2_new = self.critic2(torch.cat([states, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)

        actor_loss = -q_new.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.target_critic1, self.critic1)
        self.soft_update(self.target_critic2, self.critic2)

    def soft_update(self, target, source):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class DreamerAgent(nn.Module):
    """Dreamer agent implementation with world model and actor-critic."""

    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=256):
        super(DreamerAgent, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # World model components
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )

        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # Actor-Critic components
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean and log_std
        )

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, obs):
        """Encode observation to latent state."""
        encoded = self.encoder(obs)
        mean, log_std = encoded.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        return mean, std

    def decode(self, latent):
        """Decode latent state to observation."""
        return self.decoder(latent)

    def predict_next(self, latent, action):
        """Predict next latent state."""
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([latent, action], dim=-1)
        output = self.dynamics(input_tensor)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        return mean, std

    def predict_reward(self, latent):
        """Predict reward from latent state."""
        return self.reward_model(latent)

    def get_action(self, latent, deterministic=False):
        """Get action from latent state."""
        output = self.actor(latent)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))

        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        return action.clamp(-1, 1)

    def get_value(self, latent):
        """Get value from latent state."""
        return self.value_model(latent)


class ModelBasedMetaLearning(nn.Module):
    """Model-based meta-learning for few-shot adaptation."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, meta_lr=0.01):
        super(ModelBasedMetaLearning, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.meta_lr = meta_lr

        # Meta-model (MAML-style)
        self.meta_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1),  # next_state + reward
        )

        # Task-specific adaptation layers
        self.adaptation_layers = nn.ModuleList(
            [
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, state_dim + 1),
            ]
        )

    def forward(self, state, action, adapted_params=None):
        """Forward pass with optional adapted parameters."""
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([state, action], dim=-1)

        if adapted_params is not None:
            # Use adapted parameters
            x = F.relu(self.adaptation_layers[0](input_tensor))
            x = F.relu(self.adaptation_layers[1](x))
            output = self.adaptation_layers[2](x)
        else:
            # Use meta-model
            output = self.meta_model(input_tensor)

        next_state = output[:, : self.state_dim]
        reward = output[:, self.state_dim :]

        return next_state, reward.squeeze()

    def adapt_to_task(self, support_data, adaptation_steps=5):
        """Adapt model to specific task using support data."""
        adapted_params = []

        for layer in self.adaptation_layers:
            adapted_params.append(layer.weight.clone())
            adapted_params.append(layer.bias.clone())

        for step in range(adaptation_steps):
            # Compute gradients
            total_loss = 0
            for state, action, next_state, reward in support_data:
                pred_next_state, pred_reward = self.forward(
                    state, action, adapted_params
                )
                loss = F.mse_loss(pred_next_state, next_state) + F.mse_loss(
                    pred_reward, reward
                )
                total_loss += loss

            # Update adapted parameters
            grads = torch.autograd.grad(total_loss, adapted_params, create_graph=True)
            for i, (param, grad) in enumerate(zip(adapted_params, grads)):
                adapted_params[i] = param - self.meta_lr * grad

        return adapted_params


class SafeModelBasedRL(nn.Module):
    """Safe model-based RL with constraint satisfaction."""

    def __init__(self, state_dim, action_dim, constraint_dim, hidden_dim=256):
        super(SafeModelBasedRL, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim

        # Dynamics model
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Constraint model
        self.constraint_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, constraint_dim),
        )

        # Safety critic
        self.safety_critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Policy with safety constraints
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean and log_std
        )

        self.safety_threshold = 0.1
        self.lambda_safety = 1.0  # Safety penalty weight

    def predict_dynamics(self, state, action):
        """Predict next state."""
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([state, action], dim=-1)
        next_state = self.dynamics(input_tensor)
        return next_state

    def predict_constraints(self, state, action):
        """Predict constraint violations."""
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([state, action], dim=-1)
        constraints = self.constraint_model(input_tensor)
        return constraints

    def get_safe_action(self, state, deterministic=False):
        """Get action that satisfies safety constraints."""
        output = self.policy(state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))

        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        # Check safety constraints
        constraints = self.predict_constraints(state, action)
        safety_violations = torch.sum(
            torch.relu(constraints - self.safety_threshold), dim=-1
        )

        if torch.any(safety_violations > 0):
            # Project action to safe set
            action = self.project_to_safe_set(state, action, constraints)

        return action.clamp(-1, 1)

    def project_to_safe_set(self, state, action, constraints):
        """Project action to safe constraint set."""
        # Simple projection: reduce action magnitude if constraints violated
        violation_mask = constraints > self.safety_threshold
        if torch.any(violation_mask):
            action = action * 0.5  # Reduce action magnitude
        return action

    def compute_safety_loss(self, states, actions, next_states, constraints):
        """Compute safety loss for training."""
        pred_constraints = self.predict_constraints(states, actions)
        safety_loss = F.mse_loss(pred_constraints, constraints)
        return safety_loss


