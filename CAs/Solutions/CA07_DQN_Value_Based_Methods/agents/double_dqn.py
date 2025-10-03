"""
Double DQN Implementation
========================

This module implements Double DQN, which addresses the overestimation bias
in standard DQN by decoupling action selection from action evaluation.

Key improvements:
- Uses main network for action selection
- Uses target network for action evaluation
- Reduces overestimation bias significantly

Author: CA7 Implementation
"""

import torch
import torch.nn.functional as F
from .core import DQNAgent, device
import numpy as np


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent - Extends basic DQN to address overestimation bias

    The key innovation is decoupling action selection from action evaluation:
    - Use main network to select the best action in next state
    - Use target network to evaluate that action's value

    This prevents the overestimation bias that occurs when the same network
    is used for both selection and evaluation.
    """

    def __init__(self, state_dim, action_dim, **kwargs):
        """
        Initialize Double DQN agent

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            **kwargs: Additional arguments passed to parent DQNAgent
        """
        super().__init__(state_dim, action_dim, **kwargs)

        self.q_value_estimates = {"main": [], "target": [], "double": []}
        self.overestimation_metrics = []

    def train_step(self):
        """
        Double DQN training step with bias tracking

        The main difference from standard DQN is in target computation:
        - Standard DQN: y = r + γ * max_a' Q_target(s', a')
        - Double DQN: y = r + γ * Q_target(s', argmax_a' Q_main(s', a'))

        Returns:
            Training loss (None if insufficient buffer size)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():

            next_actions = self.q_network(next_states).argmax(1)

            next_q_values = (
                self.target_network(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )

            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

            standard_next_q_values = self.target_network(next_states).max(1)[0]
            standard_targets = rewards + (
                self.gamma * standard_next_q_values * (~dones)
            )

            self.track_bias_metrics(current_q_values, target_q_values, standard_targets)

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        self.epsilon_history.append(self.epsilon)

        with torch.no_grad():
            avg_q_value = current_q_values.mean().item()
            self.q_values_history.append(avg_q_value)

        return loss.item()

    def track_bias_metrics(self, current_q, double_targets, standard_targets):
        """
        Track overestimation bias metrics for analysis

        Args:
            current_q: Current Q-values
            double_targets: Double DQN targets
            standard_targets: Standard DQN targets
        """

        self.q_value_estimates["main"].append(current_q.mean().item())
        self.q_value_estimates["double"].append(double_targets.mean().item())

        overestimation = (standard_targets - double_targets).mean().item()
        self.overestimation_metrics.append(overestimation)

    def get_bias_statistics(self):
        """
        Get overestimation bias statistics

        Returns:
            Dictionary with bias statistics
        """
        if not self.overestimation_metrics:
            return None

        bias_array = np.array(self.overestimation_metrics)
        return {
            "mean_bias": np.mean(bias_array),
            "std_bias": np.std(bias_array),
            "max_bias": np.max(bias_array),
            "min_bias": np.min(bias_array),
            "bias_history": self.overestimation_metrics.copy(),
        }
