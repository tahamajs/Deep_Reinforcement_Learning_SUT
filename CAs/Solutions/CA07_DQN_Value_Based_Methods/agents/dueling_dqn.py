"""
Dueling DQN Implementation for CA07
====================================
This module implements Dueling DQN architecture that separates value and advantage estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .core import DQNAgent


class DuelingQNetwork(nn.Module):
    """Dueling Q-network architecture"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()

        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage: Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def get_value_and_advantage(self, x: torch.Tensor) -> tuple:
        """Get separate value and advantage estimates"""
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        return value, advantage


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN Agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace Q-network with Dueling Q-network
        self.q_network = DuelingQNetwork(
            self.state_dim, self.action_dim, kwargs.get("hidden_dim", 128)
        ).to(self.device)

        self.target_network = DuelingQNetwork(
            self.state_dim, self.action_dim, kwargs.get("hidden_dim", 128)
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Reinitialize optimizer with new parameters
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=kwargs.get("lr", 1e-3)
        )

    def get_value_and_advantage(self, state: np.ndarray) -> tuple:
        """Get value and advantage estimates for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value, advantage = self.q_network.get_value_and_advantage(state_tensor)
            return (
                float(value.cpu().numpy().flatten()[0]),
                advantage.cpu().numpy().flatten(),
            )

    def analyze_value_advantage_decomposition(
        self, env, num_samples: int = 1000
    ) -> dict:
        """Analyze the value-advantage decomposition"""
        states = []
        values = []
        advantages = []

        # Collect random states
        state, _ = env.reset()
        for _ in range(num_samples):
            states.append(state)
            action = self.select_action(state, epsilon=0.0)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                state, _ = env.reset()
            else:
                state = next_state

        # Calculate value and advantage for each state
        for state in states:
            value, advantage = self.get_value_and_advantage(state)
            values.append(value)
            advantages.append(advantage)

        values = np.array(values)
        advantages = np.array(advantages)

        return {
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "mean_advantage": np.mean(advantages),
            "std_advantage": np.std(advantages),
            "advantage_range": np.max(advantages) - np.min(advantages),
            "value_advantage_correlation": np.corrcoef(
                values, np.mean(advantages, axis=1)
            )[0, 1],
        }

    def visualize_value_advantage(self, state: np.ndarray) -> dict:
        """Visualize value and advantage components for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get Q-values
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()

            # Get value and advantage separately
            value, advantage = self.q_network.get_value_and_advantage(state_tensor)
            value = value.cpu().numpy().flatten()[0]
            advantage = advantage.cpu().numpy().flatten()

            # Reconstruct Q-values to verify
            reconstructed_q = value + advantage - np.mean(advantage)

            return {
                "state": state,
                "q_values": q_values,
                "value": value,
                "advantage": advantage,
                "reconstructed_q": reconstructed_q,
                "reconstruction_error": np.mean(np.abs(q_values - reconstructed_q)),
            }


class DuelingDoubleDQNAgent(DuelingDQNAgent):
    """Dueling Double DQN Agent combining both improvements"""

    def train_step(self) -> Optional[float]:
        """Perform one training step with Dueling Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Compute target Q values using Double DQN
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)

            # Evaluate actions using target network
            next_q_values = (
                self.target_network(next_states).gather(1, next_actions).squeeze(1)
            )
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value
