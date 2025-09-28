"""Model-Based Reinforcement Learning Agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
import random


class ModelBasedAgent:
    """Model-based RL agent using learned dynamics."""

    def __init__(self, state_dim, action_dim, lr=1e-3, planning_horizon=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.planning_horizon = planning_horizon

        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
        )

        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=lr)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        self.model_buffer = None  # Will be set to ReplayBuffer
        self.planning_buffer = None  # Will be set to ReplayBuffer

        self.model_losses = []
        self.value_losses = []

    def act(self, state, epsilon=0.1):
        """Select action using model-based planning."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        return self.plan_action(state)

    def plan_action(self, state):
        """Plan best action using learned model."""
        best_action = 0
        best_value = float("-inf")

        for action in range(self.action_dim):
            value = self.simulate_trajectory(state, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def simulate_trajectory(self, initial_state, initial_action):
        """Simulate trajectory using learned model."""
        state = torch.FloatTensor(initial_state)
        total_reward = 0.0
        gamma = 0.99

        for step in range(self.planning_horizon):
            if step == 0:
                action = initial_action
            else:
                action = self.get_greedy_action(state)

            action_tensor = torch.FloatTensor([action])
            action_one_hot = F.one_hot(action_tensor.long(), self.action_dim).float()

            model_input = torch.cat([state, action_one_hot], dim=-1)

            with torch.no_grad():
                next_state = self.dynamics_model(model_input)
                reward = self.reward_model(model_input).item()

            total_reward += (gamma**step) * reward
            state = next_state

        with torch.no_grad():
            terminal_value = self.value_network(state).item()
            total_reward += (gamma**self.planning_horizon) * terminal_value

        return total_reward

    def get_greedy_action(self, state):
        """Get greedy action for planning."""
        best_action = 0
        best_q = float("-inf")

        for action in range(self.action_dim):
            action_tensor = torch.FloatTensor([action])
            action_one_hot = F.one_hot(action_tensor.long(), self.action_dim).float()
            model_input = torch.cat([state, action_one_hot], dim=-1)

            with torch.no_grad():
                q_value = self.reward_model(model_input).item()

            if q_value > best_q:
                best_q = q_value
                best_action = action

        return best_action

    def update_model(self, batch):
        """Update dynamics and reward models."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        actions_one_hot = F.one_hot(actions, self.action_dim).float()
        model_input = torch.cat([states, actions_one_hot], dim=-1)

        pred_next_states = self.dynamics_model(model_input)
        dynamics_loss = F.mse_loss(pred_next_states, next_states)

        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        pred_rewards = self.reward_model(model_input).squeeze()
        reward_loss = F.mse_loss(pred_rewards, rewards)

        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

        total_model_loss = dynamics_loss.item() + reward_loss.item()
        self.model_losses.append(total_model_loss)

        return total_model_loss

    def update_value_function(self, batch):
        """Update value function using temporal difference learning."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_values = self.value_network(states).squeeze()

        with torch.no_grad():
            next_values = self.value_network(next_states).squeeze()
            targets = rewards + 0.99 * next_values * (~dones)

        value_loss = F.mse_loss(current_values, targets)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.value_losses.append(value_loss.item())

        return value_loss.item()


class HybridDynaAgent:
    """Dyna-Q style hybrid agent combining model-free and model-based learning."""

    def __init__(self, state_dim, action_dim, lr=1e-3, planning_steps=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_steps = planning_steps

        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.lr = lr
        self.gamma = 0.99

        self.model = {}  # (s,a) -> (r, s', done)
        self.visited_states = set()
        self.experience_buffer = deque(maxlen=10000)

    def act(self, state, epsilon=0.1):
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        state_key = tuple(state) if isinstance(state, np.ndarray) else state
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        """Dyna-Q update: direct RL + model learning + planning."""
        state_key = tuple(state) if isinstance(state, np.ndarray) else state
        next_state_key = (
            tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
        )

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        self.q_table[state_key][action] += self.lr * (
            target - self.q_table[state_key][action]
        )

        self.model[(state_key, action)] = (reward, next_state_key, done)
        self.visited_states.add(state_key)
        self.experience_buffer.append((state_key, action, reward, next_state_key, done))

        self.planning_updates()

    def planning_updates(self):
        """Perform planning updates using learned model."""
        if len(self.experience_buffer) == 0:
            return

        for _ in range(self.planning_steps):
            if len(self.experience_buffer) > 0:
                state_key, action, reward, next_state_key, done = random.choice(
                    self.experience_buffer
                )

                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.q_table[next_state_key])

                self.q_table[state_key][action] += self.lr * (
                    target - self.q_table[state_key][action]
                )
