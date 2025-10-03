import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DynamicsModel(nn.Module):
    """Neural network for predicting environment dynamics"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Dynamics network: (state, action) -> next_state
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action):
        # Convert action to one-hot encoding
        if action.dim() == 1:
            action_onehot = F.one_hot(action, self.action_dim).float()
        else:
            action_onehot = action.float()
            
        x = torch.cat([state, action_onehot], dim=-1)
        return self.network(x)


class RewardModel(nn.Module):
    """Neural network for predicting rewards"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(RewardModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Reward network: (state, action) -> reward
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        # Convert action to one-hot encoding
        if action.dim() == 1:
            action_onehot = F.one_hot(action, self.action_dim).float()
        else:
            action_onehot = action.float()
            
        x = torch.cat([state, action_onehot], dim=-1)
        return self.network(x)


class ModelBasedAgent:
    """Model-based RL agent that learns environment dynamics"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.dynamics_model = DynamicsModel(state_dim, action_dim, hidden_dim).to(self.device)
        self.reward_model = RewardModel(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)
        
        # Optimizers
        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Planning parameters
        self.planning_horizon = 5
        self.num_rollouts = 10
        
    def act(self, state, epsilon=0.1):
        """Select action using model-based planning"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        # Use Q-network for action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def plan(self, state):
        """Plan using learned model"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        best_action = 0
        best_value = float('-inf')
        
        for action in range(self.action_dim):
            total_value = 0.0
            action_tensor = torch.LongTensor([action])
            
            # Simulate multiple rollouts
            for _ in range(self.num_rollouts):
                current_state = state_tensor.clone()
                rollout_value = 0.0
                
                for step in range(self.planning_horizon):
                    # Predict next state
                    next_state = self.dynamics_model(current_state, action_tensor)
                    
                    # Predict reward
                    reward = self.reward_model(current_state, action_tensor)
                    rollout_value += (self.gamma ** step) * reward.item()
                    
                    # Update state for next step
                    current_state = next_state
                    
                    # Select next action (can be random or from policy)
                    next_action = random.randint(0, self.action_dim - 1)
                    action_tensor = torch.LongTensor([next_action])
                
                total_value += rollout_value
            
            avg_value = total_value / self.num_rollouts
            if avg_value > best_value:
                best_value = avg_value
                best_action = action
                
        return best_action
    
    def update(self, batch_size=32):
        """Update models using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return
            
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        
        # Update dynamics model
        predicted_next_states = self.dynamics_model(states, actions)
        dynamics_loss = F.mse_loss(predicted_next_states, next_states)
        
        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()
        
        # Update reward model
        predicted_rewards = self.reward_model(states, actions).squeeze()
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()
        
        # Update Q-network using model predictions
        with torch.no_grad():
            # Use predicted next states for target computation
            next_q_values = self.q_network(predicted_next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return {
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item(),
            'q_loss': q_loss.item()
        }
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))