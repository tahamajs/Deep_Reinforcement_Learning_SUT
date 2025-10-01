"""
Continual Learning Agent

This module implements a continual learning agent that can learn multiple tasks sequentially.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from .ewc import ElasticWeightConsolidation, EWCNetwork
from .progressive_networks import ProgressiveNetwork
from .experience_replay import ExperienceReplay, ContinualExperienceReplay


class ContinualLearningAgent:
    """Agent that can learn multiple tasks sequentially without forgetting."""
    
    def __init__(self, state_dim: int, action_dim: int, method: str = 'ewc', 
                 lr: float = 1e-3, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.method = method
        self.device = device
        
        # Initialize network based on method
        if method == 'ewc':
            self.policy = EWCNetwork(state_dim, action_dim)
            self.policy.setup_ewc(device=device)
        elif method == 'progressive':
            self.policy = ProgressiveNetwork(state_dim, action_dim)
        else:
            # Standard network
            self.policy = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience replay
        self.experience_replay = ContinualExperienceReplay(device=device)
        
        # Task tracking
        self.current_task = 0
        self.task_performances = {}
        self.training_history = {
            'losses': [],
            'rewards': [],
            'task_transitions': []
        }
    
    def select_action(self, state: torch.Tensor, task_id: int = None) -> int:
        """Select action for current or specified task."""
        if task_id is None:
            task_id = self.current_task
        
        with torch.no_grad():
            if self.method == 'progressive':
                action_logits = self.policy(state.unsqueeze(0), task_id)
            else:
                action_logits = self.policy(state.unsqueeze(0))
            
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1)
            
            return action.item()
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
               rewards: torch.Tensor, task_id: int = None) -> float:
        """Update policy for current task."""
        if task_id is None:
            task_id = self.current_task
        
        # Forward pass
        if self.method == 'progressive':
            action_logits = self.policy(states, task_id)
        else:
            action_logits = self.policy(states)
        
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        policy_loss = -(log_probs * rewards.unsqueeze(1)).mean()
        
        # Add regularization based on method
        if self.method == 'ewc':
            ewc_loss = self.policy.ewc.compute_ewc_loss(task_id)
            total_loss = policy_loss + ewc_loss
        else:
            total_loss = policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Record training history
        self.training_history['losses'].append(total_loss.item())
        self.training_history['rewards'].append(rewards.mean().item())
        
        return total_loss.item()
    
    def add_task(self, task_id: int, output_dim: int = None):
        """Add a new task to the agent."""
        if self.method == 'progressive':
            self.policy.add_task_column(task_id, output_dim)
        
        # Initialize task performance tracking
        self.task_performances[task_id] = []
        
        # Record task transition
        self.training_history['task_transitions'].append({
            'task_id': task_id,
            'timestamp': len(self.training_history['losses'])
        })
    
    def switch_task(self, task_id: int):
        """Switch to a different task."""
        if task_id not in self.task_performances:
            self.add_task(task_id)
        
        self.current_task = task_id
    
    def evaluate_task(self, test_data: List[Tuple], task_id: int) -> Dict[str, float]:
        """Evaluate performance on a specific task."""
        if task_id not in self.task_performances:
            raise ValueError(f"Task {task_id} not found.")
        
        total_reward = 0.0
        num_episodes = len(test_data)
        
        for states, actions, rewards in test_data:
            episode_reward = 0.0
            for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
                # Get predicted action
                predicted_action = self.select_action(state, task_id)
                
                # Calculate reward (simplified)
                if predicted_action == action:
                    episode_reward += reward
                else:
                    episode_reward += reward * 0.5  # Partial credit
            
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        self.task_performances[task_id].append(avg_reward)
        
        return {'avg_reward': avg_reward, 'num_episodes': num_episodes}
    
    def compute_forgetting_measure(self, task_id: int) -> float:
        """Compute forgetting measure for a task."""
        if task_id not in self.task_performances or len(self.task_performances[task_id]) < 2:
            return 0.0
        
        performances = self.task_performances[task_id]
        initial_performance = performances[0]
        final_performance = performances[-1]
        
        return max(0.0, initial_performance - final_performance)
    
    def compute_transfer_measure(self, task_id: int) -> float:
        """Compute transfer measure for a task."""
        if task_id not in self.task_performances or len(self.task_performances[task_id]) < 2:
            return 0.0
        
        performances = self.task_performances[task_id]
        initial_performance = performances[0]
        final_performance = performances[-1]
        
        return max(0.0, final_performance - initial_performance)
    
    def get_task_statistics(self, task_id: int) -> Dict[str, Any]:
        """Get statistics for a specific task."""
        if task_id not in self.task_performances:
            return {}
        
        stats = {
            'task_id': task_id,
            'performances': self.task_performances[task_id].copy(),
            'forgetting_measure': self.compute_forgetting_measure(task_id),
            'transfer_measure': self.compute_transfer_measure(task_id),
            'num_evaluations': len(self.task_performances[task_id])
        }
        
        if self.method == 'progressive':
            stats.update(self.policy.get_column_statistics(task_id))
        elif self.method == 'ewc':
            ewc_stats = self.policy.get_ewc_statistics()
            stats['ewc_losses'] = ewc_stats.get('ewc_losses', [])
        
        return stats
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall agent statistics."""
        stats = {
            'method': self.method,
            'current_task': self.current_task,
            'num_tasks': len(self.task_performances),
            'task_performances': {k: v.copy() for k, v in self.task_performances.items()},
            'training_history': {
                'losses': self.training_history['losses'].copy(),
                'rewards': self.training_history['rewards'].copy(),
                'task_transitions': self.training_history['task_transitions'].copy()
            }
        }
        
        # Add method-specific statistics
        if self.method == 'progressive':
            stats['network_statistics'] = self.policy.get_network_statistics()
        elif self.method == 'ewc':
            stats['ewc_statistics'] = self.policy.get_ewc_statistics()
        
        # Add experience replay statistics
        stats['experience_replay'] = self.experience_replay.get_statistics()
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the model to file."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_task': self.current_task,
            'task_performances': self.task_performances,
            'training_history': self.training_history,
            'method': self.method
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_task = checkpoint['current_task']
        self.task_performances = checkpoint['task_performances']
        self.training_history = checkpoint['training_history']
        self.method = checkpoint['method']


class LifelongLearner(ContinualLearningAgent):
    """Extended continual learning agent with lifelong learning capabilities."""
    
    def __init__(self, state_dim: int, action_dim: int, method: str = 'ewc', 
                 lr: float = 1e-3, device: str = 'cpu'):
        super().__init__(state_dim, action_dim, method, lr, device)
        
        # Lifelong learning specific components
        self.task_importance_weights = {}
        self.knowledge_transfer_matrix = {}
        self.adaptation_history = {}
        
        # Meta-learning components
        self.meta_learning_rate = 1e-4
        self.adaptation_steps = 5
    
    def compute_task_importance(self, task_id: int) -> float:
        """Compute importance weight for a task."""
        if task_id not in self.task_performances:
            return 1.0
        
        performances = self.task_performances[task_id]
        if not performances:
            return 1.0
        
        # Importance based on performance stability and recent performance
        recent_performance = performances[-1] if performances else 0.0
        performance_stability = 1.0 - np.std(performances) if len(performances) > 1 else 1.0
        
        importance = recent_performance * performance_stability
        self.task_importance_weights[task_id] = importance
        
        return importance
    
    def update_knowledge_transfer(self, source_task: int, target_task: int, 
                                transfer_strength: float):
        """Update knowledge transfer matrix."""
        if source_task not in self.knowledge_transfer_matrix:
            self.knowledge_transfer_matrix[source_task] = {}
        
        self.knowledge_transfer_matrix[source_task][target_task] = transfer_strength
    
    def adapt_to_new_task(self, task_id: int, adaptation_data: List[Tuple], 
                         num_adaptation_steps: int = None) -> float:
        """Adapt to a new task using meta-learning."""
        if num_adaptation_steps is None:
            num_adaptation_steps = self.adaptation_steps
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.policy.named_parameters()}
        
        # Adaptation steps
        adaptation_losses = []
        for step in range(num_adaptation_steps):
            step_loss = 0.0
            num_batches = 0
            
            for states, actions, rewards in adaptation_data:
                # Forward pass
                if self.method == 'progressive':
                    action_logits = self.policy(states, task_id)
                else:
                    action_logits = self.policy(states)
                
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
                loss = -(log_probs * rewards.unsqueeze(1)).mean()
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                step_loss += loss.item()
                num_batches += 1
            
            avg_loss = step_loss / num_batches
            adaptation_losses.append(avg_loss)
        
        # Record adaptation history
        self.adaptation_history[task_id] = {
            'losses': adaptation_losses,
            'num_steps': num_adaptation_steps,
            'final_loss': adaptation_losses[-1] if adaptation_losses else 0.0
        }
        
        return adaptation_losses[-1] if adaptation_losses else 0.0
    
    def get_lifelong_statistics(self) -> Dict[str, Any]:
        """Get lifelong learning statistics."""
        base_stats = self.get_overall_statistics()
        
        lifelong_stats = {
            'task_importance_weights': self.task_importance_weights.copy(),
            'knowledge_transfer_matrix': self.knowledge_transfer_matrix.copy(),
            'adaptation_history': self.adaptation_history.copy()
        }
        
        # Compute lifelong learning metrics
        if len(self.task_performances) > 1:
            # Average forgetting across all tasks
            forgetting_measures = [self.compute_forgetting_measure(task_id) 
                                 for task_id in self.task_performances.keys()]
            lifelong_stats['avg_forgetting'] = np.mean(forgetting_measures)
            
            # Average transfer across all tasks
            transfer_measures = [self.compute_transfer_measure(task_id) 
                               for task_id in self.task_performances.keys()]
            lifelong_stats['avg_transfer'] = np.mean(transfer_measures)
        
        base_stats['lifelong_learning'] = lifelong_stats
        return base_stats


class TransferLearningAgent(ContinualLearningAgent):
    """Agent specialized for transfer learning between tasks."""
    
    def __init__(self, state_dim: int, action_dim: int, method: str = 'ewc', 
                 lr: float = 1e-3, device: str = 'cpu'):
        super().__init__(state_dim, action_dim, method, lr, device)
        
        # Transfer learning specific components
        self.transfer_weights = {}
        self.similarity_matrix = {}
        self.transfer_history = {}
    
    def compute_task_similarity(self, task1: int, task2: int) -> float:
        """Compute similarity between two tasks."""
        if task1 not in self.task_performances or task2 not in self.task_performances:
            return 0.0
        
        # Simple similarity based on performance patterns
        perf1 = self.task_performances[task1]
        perf2 = self.task_performances[task2]
        
        if not perf1 or not perf2:
            return 0.0
        
        # Compute correlation between performance patterns
        min_len = min(len(perf1), len(perf2))
        if min_len < 2:
            return 0.0
        
        corr = np.corrcoef(perf1[:min_len], perf2[:min_len])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def transfer_knowledge(self, source_task: int, target_task: int, 
                          transfer_strength: float = 0.5) -> float:
        """Transfer knowledge from source task to target task."""
        if source_task not in self.task_performances or target_task not in self.task_performances:
            return 0.0
        
        # Compute similarity
        similarity = self.compute_task_similarity(source_task, target_task)
        
        # Apply transfer
        transfer_effect = similarity * transfer_strength
        
        # Record transfer
        if source_task not in self.transfer_history:
            self.transfer_history[source_task] = {}
        
        self.transfer_history[source_task][target_task] = {
            'similarity': similarity,
            'transfer_strength': transfer_strength,
            'transfer_effect': transfer_effect
        }
        
        return transfer_effect
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get transfer learning statistics."""
        stats = {
            'transfer_weights': self.transfer_weights.copy(),
            'similarity_matrix': self.similarity_matrix.copy(),
            'transfer_history': self.transfer_history.copy()
        }
        
        # Compute transfer metrics
        if len(self.task_performances) > 1:
            task_ids = list(self.task_performances.keys())
            similarities = []
            
            for i, task1 in enumerate(task_ids):
                for j, task2 in enumerate(task_ids):
                    if i != j:
                        similarity = self.compute_task_similarity(task1, task2)
                        similarities.append(similarity)
            
            if similarities:
                stats['avg_task_similarity'] = np.mean(similarities)
                stats['max_task_similarity'] = np.max(similarities)
                stats['min_task_similarity'] = np.min(similarities)
        
        return stats
