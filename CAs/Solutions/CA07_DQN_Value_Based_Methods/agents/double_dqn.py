"""
Double DQN Implementation for CA07
==================================
This module implements Double DQN to reduce overestimation bias in Q-learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .core import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent to reduce overestimation bias"""

    def train_step(self) -> Optional[float]:
        """Perform one training step with Double DQN"""
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

    def get_action_selection_bias(self, states: torch.Tensor) -> torch.Tensor:
        """Calculate the bias in action selection (for analysis)"""
        with torch.no_grad():
            # Q-values from online network
            online_q_values = self.q_network(states)

            # Q-values from target network
            target_q_values = self.target_network(states)

            # Bias is the difference between max Q-values
            online_max = online_q_values.max(1)[0]
            target_max = target_q_values.max(1)[0]

            return online_max - target_max

    def analyze_overestimation_bias(self, env, num_samples: int = 1000) -> dict:
        """Analyze overestimation bias in the current policy"""
        states = []
        biases = []

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

        # Calculate biases
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        biases = self.get_action_selection_bias(states_tensor)

        return {
            "mean_bias": biases.mean().item(),
            "std_bias": biases.std().item(),
            "max_bias": biases.max().item(),
            "min_bias": biases.min().item(),
            "positive_bias_ratio": (biases > 0).float().mean().item(),
        }
