"""
Experience Replay for Continual Learning

This module implements various experience replay strategies for continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import random
import heapq
from dataclasses import dataclass


@dataclass
class Experience:
    """Represents a single experience."""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    task_id: int
    priority: float = 1.0
    timestamp: float = 0.0


class ExperienceReplay:
    """Standard experience replay buffer."""
    
    def __init__(self, capacity: int = 10000, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.task_experiences = {}  # Maps task_id to list of indices
        
        # Statistics
        self.stats = {
            'total_experiences': 0,
            'task_counts': {},
            'avg_reward': 0.0
        }
    
    def add(self, experience: Experience):
        """Add experience to buffer."""
        # Update task tracking
        if experience.task_id not in self.task_experiences:
            self.task_experiences[experience.task_id] = []
        
        # Add to buffer
        self.buffer.append(experience)
        
        # Update task experiences (store index)
        buffer_idx = len(self.buffer) - 1
        self.task_experiences[experience.task_id].append(buffer_idx)
        
        # Update statistics
        self.stats['total_experiences'] += 1
        if experience.task_id not in self.stats['task_counts']:
            self.stats['task_counts'][experience.task_id] = 0
        self.stats['task_counts'][experience.task_id] += 1
        
        # Update average reward
        total_reward = self.stats['avg_reward'] * (self.stats['total_experiences'] - 1)
        self.stats['avg_reward'] = (total_reward + experience.reward) / self.stats['total_experiences']
    
    def sample(self, batch_size: int, task_id: int = None) -> List[Experience]:
        """Sample experiences from buffer."""
        if task_id is not None and task_id in self.task_experiences:
            # Sample from specific task
            task_indices = self.task_experiences[task_id]
            if len(task_indices) < batch_size:
                # If not enough experiences for this task, sample from all
                return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
            else:
                # Sample from task-specific experiences
                sampled_indices = random.sample(task_indices, batch_size)
                return [self.buffer[idx] for idx in sampled_indices if idx < len(self.buffer)]
        else:
            # Sample from all experiences
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def get_task_experiences(self, task_id: int) -> List[Experience]:
        """Get all experiences for a specific task."""
        if task_id not in self.task_experiences:
            return []
        
        task_indices = self.task_experiences[task_id]
        return [self.buffer[idx] for idx in task_indices if idx < len(self.buffer)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return self.stats.copy()
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.task_experiences.clear()
        self.stats = {
            'total_experiences': 0,
            'task_counts': {},
            'avg_reward': 0.0
        }


class PrioritizedExperienceReplay:
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, 
                 device: str = 'cpu'):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling correction
        self.device = device
        
        # Priority queue (heap)
        self.priorities = []
        self.experiences = {}
        self.max_priority = 1.0
        
        # Statistics
        self.stats = {
            'total_experiences': 0,
            'task_counts': {},
            'avg_priority': 0.0
        }
    
    def add(self, experience: Experience):
        """Add experience with priority."""
        # Calculate priority
        priority = self.max_priority ** self.alpha
        
        # Add to priority queue
        heapq.heappush(self.priorities, (-priority, self.stats['total_experiences']))
        
        # Store experience
        self.experiences[self.stats['total_experiences']] = experience
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Update statistics
        self.stats['total_experiences'] += 1
        if experience.task_id not in self.stats['task_counts']:
            self.stats['task_counts'][experience.task_id] = 0
        self.stats['task_counts'][experience.task_id] += 1
        
        # Update average priority
        total_priority = self.stats['avg_priority'] * (self.stats['total_experiences'] - 1)
        self.stats['avg_priority'] = (total_priority + priority) / self.stats['total_experiences']
        
        # Maintain capacity
        if len(self.priorities) > self.capacity:
            # Remove lowest priority experience
            _, oldest_idx = heapq.heappop(self.priorities)
            if oldest_idx in self.experiences:
                del self.experiences[oldest_idx]
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample experiences with priorities and importance weights."""
        if len(self.priorities) < batch_size:
            batch_size = len(self.priorities)
        
        # Sample based on priorities
        sampled_experiences = []
        sampled_priorities = []
        sampled_indices = []
        
        # Get top priorities
        top_priorities = heapq.nlargest(batch_size, self.priorities)
        
        for neg_priority, idx in top_priorities:
            if idx in self.experiences:
                priority = -neg_priority
                experience = self.experiences[idx]
                
                sampled_experiences.append(experience)
                sampled_priorities.append(priority)
                sampled_indices.append(idx)
        
        # Calculate importance weights
        weights = []
        for priority in sampled_priorities:
            # Importance sampling weight: (N * P(i))^(-beta) / max_weight
            weight = (len(self.priorities) * priority) ** (-self.beta)
            weights.append(weight)
        
        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]
        
        return sampled_experiences, weights, sampled_indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for specific experiences."""
        for idx, priority in zip(indices, priorities):
            if idx in self.experiences:
                # Update max priority
                self.max_priority = max(self.max_priority, priority)
                
                # Add new priority to queue
                heapq.heappush(self.priorities, (-priority, idx))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return self.stats.copy()


class ContinualExperienceReplay:
    """Experience replay specifically designed for continual learning."""
    
    def __init__(self, capacity: int = 10000, task_capacity: int = 1000, 
                 device: str = 'cpu'):
        self.capacity = capacity
        self.task_capacity = task_capacity
        self.device = device
        
        # Task-specific buffers
        self.task_buffers = {}
        self.global_buffer = deque(maxlen=capacity)
        
        # Task importance weights
        self.task_weights = {}
        
        # Statistics
        self.stats = {
            'total_experiences': 0,
            'task_counts': {},
            'task_weights': {},
            'forgetting_measures': {}
        }
    
    def add(self, experience: Experience):
        """Add experience to appropriate buffers."""
        task_id = experience.task_id
        
        # Add to task-specific buffer
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = deque(maxlen=self.task_capacity)
        
        self.task_buffers[task_id].append(experience)
        
        # Add to global buffer
        self.global_buffer.append(experience)
        
        # Update statistics
        self.stats['total_experiences'] += 1
        if task_id not in self.stats['task_counts']:
            self.stats['task_counts'][task_id] = 0
        self.stats['task_counts'][task_id] += 1
    
    def sample(self, batch_size: int, task_id: int = None, 
               use_task_weights: bool = True) -> List[Experience]:
        """Sample experiences with task weighting."""
        if task_id is not None and task_id in self.task_buffers:
            # Sample from specific task
            task_experiences = list(self.task_buffers[task_id])
            if len(task_experiences) >= batch_size:
                return random.sample(task_experiences, batch_size)
            else:
                # Supplement with global experiences
                remaining = batch_size - len(task_experiences)
                global_sample = random.sample(list(self.global_buffer), 
                                            min(remaining, len(self.global_buffer)))
                return task_experiences + global_sample
        
        # Sample from global buffer with task weighting
        if use_task_weights and self.task_weights:
            return self._weighted_sample(batch_size)
        else:
            return random.sample(list(self.global_buffer), 
                               min(batch_size, len(self.global_buffer)))
    
    def _weighted_sample(self, batch_size: int) -> List[Experience]:
        """Sample experiences with task weighting."""
        # Calculate sampling probabilities for each task
        task_probs = {}
        total_weight = sum(self.task_weights.values())
        
        for task_id, weight in self.task_weights.items():
            if task_id in self.task_buffers and len(self.task_buffers[task_id]) > 0:
                task_probs[task_id] = weight / total_weight
        
        # Sample experiences
        sampled_experiences = []
        for _ in range(batch_size):
            # Choose task based on weights
            task_id = np.random.choice(list(task_probs.keys()), 
                                     p=list(task_probs.values()))
            
            # Sample from chosen task
            if task_id in self.task_buffers and len(self.task_buffers[task_id]) > 0:
                experience = random.choice(list(self.task_buffers[task_id]))
                sampled_experiences.append(experience)
        
        return sampled_experiences
    
    def update_task_weights(self, task_id: int, weight: float):
        """Update importance weight for a task."""
        self.task_weights[task_id] = weight
        self.stats['task_weights'][task_id] = weight
    
    def compute_forgetting_measure(self, task_id: int, current_performance: float) -> float:
        """Compute forgetting measure for a task."""
        if task_id not in self.stats['forgetting_measures']:
            self.stats['forgetting_measures'][task_id] = []
        
        # Store current performance
        self.stats['forgetting_measures'][task_id].append(current_performance)
        
        # Calculate forgetting (simplified)
        if len(self.stats['forgetting_measures'][task_id]) > 1:
            initial_performance = self.stats['forgetting_measures'][task_id][0]
            forgetting = max(0.0, initial_performance - current_performance)
            return forgetting
        
        return 0.0
    
    def get_task_experiences(self, task_id: int) -> List[Experience]:
        """Get all experiences for a specific task."""
        if task_id not in self.task_buffers:
            return []
        return list(self.task_buffers[task_id])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return self.stats.copy()
    
    def clear_task(self, task_id: int):
        """Clear experiences for a specific task."""
        if task_id in self.task_buffers:
            del self.task_buffers[task_id]
        if task_id in self.task_weights:
            del self.task_weights[task_id]
        if task_id in self.stats['task_counts']:
            del self.stats['task_counts'][task_id]
        if task_id in self.stats['forgetting_measures']:
            del self.stats['forgetting_measures'][task_id]
    
    def clear(self):
        """Clear all buffers."""
        self.task_buffers.clear()
        self.global_buffer.clear()
        self.task_weights.clear()
        self.stats = {
            'total_experiences': 0,
            'task_counts': {},
            'task_weights': {},
            'forgetting_measures': {}
        }
