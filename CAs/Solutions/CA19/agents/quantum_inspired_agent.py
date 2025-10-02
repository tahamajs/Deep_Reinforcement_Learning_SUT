import numpy as np
import torch
import torch.nn as nn
from typing import List

class QuantumInspiredAgent:
    """
    A quantum-inspired reinforcement learning agent that uses classical neural networks
    with quantum-inspired processing principles.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.episode_rewards = []

    def parameters(self):
        return list(self.state_encoder.parameters()) + list(self.action_head.parameters())

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            encoded_state = self.state_encoder(state_tensor)
            action_probs = self.action_head(encoded_state)
            action = torch.multinomial(action_probs.squeeze(), 1).item()
        return action

    def update(self, states: List[np.ndarray], actions: List[int], rewards: List[float]):
        if not states:
            return
        self.optimizer.zero_grad()
        total_loss = 0
        for state, action, reward in zip(states, actions, rewards):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            encoded_state = self.state_encoder(state_tensor)
            action_probs = self.action_head(encoded_state)
            log_prob = torch.log(action_probs[0, action] + 1e-8)
            loss = -log_prob * reward
            total_loss += loss
        total_loss.backward()
        self.optimizer.step()
        self.episode_rewards.extend(rewards)
