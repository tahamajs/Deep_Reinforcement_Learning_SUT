"""Sample efficiency techniques for reinforcement learning."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from ..buffers.replay_buffer import PrioritizedReplayBuffer


class DataAugmentationDQN(nn.Module):
    """DQN with data augmentation and auxiliary tasks for sample efficiency."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.next_state_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

    def forward(self, state, action=None):
        """Forward pass with optional auxiliary predictions."""
        q_values = self.q_network(state)

        if action is not None:
            if len(action.shape) == 1:
                action_one_hot = F.one_hot(action.long(), self.action_dim).float()
            else:
                action_one_hot = action

            aux_input = torch.cat([state, action_one_hot], dim=-1)
            reward_pred = self.reward_predictor(aux_input)
            next_state_pred = self.next_state_predictor(aux_input)

            return q_values, reward_pred, next_state_pred

        return q_values

    def apply_augmentation(self, state, augmentation_type="noise"):
        """Apply data augmentation to state."""
        if augmentation_type == "noise":
            noise = torch.randn_like(state) * 0.1
            return state + noise

        elif augmentation_type == "dropout":
            dropout_mask = torch.rand_like(state) > 0.1
            return state * dropout_mask.float()

        elif augmentation_type == "scaling":
            scale = torch.rand(1).item() * 0.4 + 0.8  # Scale between 0.8 and 1.2
            return state * scale

        return state


class SampleEfficientAgent:
    """Agent with multiple sample efficiency techniques."""

    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = DataAugmentationDQN(state_dim, action_dim)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)

        self.gamma = 0.99
        self.target_update_freq = 100
        self.update_count = 0

        self.aux_reward_weight = 0.1
        self.aux_dynamics_weight = 0.1

        self.losses = []
        self.td_errors = []

    def act(self, state, epsilon=0.1):
        """Select action with epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network(state_tensor)
            return q_values.argmax().item()

    def update(self, batch_size=32, use_aux_tasks=True, augmentation=True):
        """Update agent with prioritized replay and auxiliary tasks."""
        sample_result = self.replay_buffer.sample(batch_size)
        if sample_result is None:
            return None

        experiences, indices, weights = sample_result
        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        weights = torch.FloatTensor(weights)

        if augmentation:
            aug_type = np.random.choice(["noise", "dropout", "scaling"])
            states = self.network.apply_augmentation(states, aug_type)
            next_states = self.network.apply_augmentation(next_states, aug_type)

        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        td_errors = (current_q_values.squeeze() - target_q_values).detach().numpy()

        q_loss = (
            weights
            * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction="none")
        ).mean()

        total_loss = q_loss

        if use_aux_tasks:
            q_values, reward_pred, next_state_pred = self.network(states, actions)

            aux_reward_loss = F.mse_loss(reward_pred.squeeze(), rewards)
            aux_dynamics_loss = F.mse_loss(next_state_pred, next_states)

            total_loss += self.aux_reward_weight * aux_reward_loss
            total_loss += self.aux_dynamics_weight * aux_dynamics_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors)

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.losses.append(total_loss.item())
        self.td_errors.extend(td_errors.tolist())

        return {
            "total_loss": total_loss.item(),
            "q_loss": q_loss.item(),
            "aux_reward_loss": aux_reward_loss.item() if use_aux_tasks else 0,
            "aux_dynamics_loss": aux_dynamics_loss.item() if use_aux_tasks else 0,
        }
