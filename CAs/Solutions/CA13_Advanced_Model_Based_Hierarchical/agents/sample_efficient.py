import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import math


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, td_error=None):
        """Add experience with priority"""
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        # Set priority based on TD error if provided
        priority = td_error if td_error is not None else max_priority
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return None
            
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get batch
        batch = [self.buffer[idx] for idx in indices]
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'indices': indices,
            'weights': weights
        }
    
    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled experiences"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
            
    def __len__(self):
        return len(self.buffer)


class SampleEfficientNetwork(nn.Module):
    """Network with auxiliary tasks for sample efficiency"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SampleEfficientNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Q-value head
        self.q_head = nn.Linear(hidden_dim, action_dim)
        
        # Auxiliary task heads
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, state, action=None):
        features = self.feature_extractor(state)
        
        q_values = self.q_head(features)
        reward_pred = self.reward_head(features)
        next_state_pred = self.next_state_head(features)
        
        if action is not None:
            return q_values, reward_pred, next_state_pred
        return q_values
    
    def apply_augmentation(self, state, aug_type='noise'):
        """Apply data augmentation to states"""
        if aug_type == 'noise':
            noise = torch.randn_like(state) * 0.1
            return state + noise
        elif aug_type == 'dropout':
            mask = torch.rand_like(state) > 0.1
            return state * mask.float()
        elif aug_type == 'scaling':
            scale = torch.rand_like(state) * 0.2 + 0.9  # Random scale between 0.9 and 1.1
            return state * scale
        else:
            return state


class SampleEfficientAgent:
    """Sample-efficient agent with prioritized replay, data augmentation, and auxiliary tasks"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99,
                 buffer_size=10000, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.network = SampleEfficientNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = SampleEfficientNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Prioritized replay buffer
        from ..buffers.replay_buffer import PrioritizedReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training parameters
        self.target_update_freq = target_update_freq
        self.training_step = 0
        
        # Loss weights for auxiliary tasks
        self.q_weight = 1.0
        self.reward_weight = 0.1
        self.dynamics_weight = 0.1
        
    def act(self, state, epsilon=0.1):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size=32):
        """Update with prioritized replay and auxiliary tasks"""
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return
            
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        weights = torch.FloatTensor(batch['weights']).to(self.device)
        
        # Forward pass with auxiliary tasks
        q_values, reward_pred, next_state_pred = self.network(states, actions)
        
        # Q-learning loss
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        td_errors = target_q_values - current_q_values
        q_loss = (td_errors ** 2 * weights).mean()
        
        # Auxiliary task losses
        reward_loss = F.mse_loss(reward_pred.squeeze(), rewards)
        dynamics_loss = F.mse_loss(next_state_pred, next_states)
        
        # Combined loss
        total_loss = (self.q_weight * q_loss + 
                     self.reward_weight * reward_loss + 
                     self.dynamics_weight * dynamics_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(batch['indices'], td_errors.detach().cpu().numpy())
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            
        self.training_step += 1
        
        return {
            'q_loss': q_loss.item(),
            'reward_loss': reward_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def store_experience(self, state, action, reward, next_state, done, td_error=None):
        """Store experience with optional TD error for priority"""
        self.replay_buffer.push(state, action, reward, next_state, done, td_error)