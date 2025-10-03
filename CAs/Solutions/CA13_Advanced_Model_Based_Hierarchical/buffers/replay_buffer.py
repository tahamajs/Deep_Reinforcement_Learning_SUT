import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Standard experience replay buffer"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
    def __len__(self):
        return len(self.buffer)


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


class NStepReplayBuffer(ReplayBuffer):
    """N-step experience replay buffer"""
    
    def __init__(self, capacity, n_step=3, gamma=0.99):
        super().__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience and compute n-step returns"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Compute n-step return
            n_step_return = 0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break
            
            # Get the first experience
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            last_next_state, _, _, last_next_state_actual, last_done = self.n_step_buffer[-1]
            
            # Store n-step experience
            super().push(first_state, first_action, n_step_return, 
                        last_next_state_actual, last_done)
    
    def sample(self, batch_size):
        """Sample batch of n-step experiences"""
        return super().sample(batch_size)


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent environments"""
    
    def __init__(self, capacity, n_agents):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffers = [deque(maxlen=capacity) for _ in range(n_agents)]
        
    def push(self, states, actions, rewards, next_states, dones):
        """Add experiences for all agents"""
        for i in range(self.n_agents):
            self.buffers[i].append((
                states[i], actions[i], rewards[i], 
                next_states[i], dones
            ))
    
    def sample(self, batch_size):
        """Sample batch for all agents"""
        if len(self.buffers[0]) < batch_size:
            return None
        
        batch = {}
        for i in range(self.n_agents):
            agent_batch = random.sample(self.buffers[i], batch_size)
            
            batch[f'states_{i}'] = np.array([e[0] for e in agent_batch])
            batch[f'actions_{i}'] = np.array([e[1] for e in agent_batch])
            batch[f'rewards_{i}'] = np.array([e[2] for e in agent_batch])
            batch[f'next_states_{i}'] = np.array([e[3] for e in agent_batch])
            batch[f'dones_{i}'] = np.array([e[4] for e in agent_batch])
        
        return batch
    
    def __len__(self):
        return len(self.buffers[0])


class EpisodeReplayBuffer:
    """Buffer that stores complete episodes"""
    
    def __init__(self, max_episodes=1000):
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode = []
        
    def add_step(self, state, action, reward, next_state, done):
        """Add step to current episode"""
        self.current_episode.append((state, action, reward, next_state, done))
        
        if done:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
    
    def sample_episodes(self, n_episodes):
        """Sample complete episodes"""
        if len(self.episodes) < n_episodes:
            return self.episodes
        return random.sample(self.episodes, n_episodes)
    
    def sample_steps(self, batch_size):
        """Sample random steps from all episodes"""
        all_steps = []
        for episode in self.episodes:
            all_steps.extend(episode)
        
        if len(all_steps) < batch_size:
            return None
        
        batch = random.sample(all_steps, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self):
        return len(self.episodes)