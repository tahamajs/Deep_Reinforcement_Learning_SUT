"""
Federated Reinforcement Learning

This module implements federated learning approaches for distributed RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import copy
import time


class FederatedClient:
    """Federated learning client for RL."""
    
    def __init__(self, client_id: str, state_dim: int, action_dim: int, 
                 lr: float = 1e-3, device: str = 'cpu'):
        self.client_id = client_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Local model
        self.local_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)
        
        # Local data
        self.local_data = []
        self.local_data_size = 0
        
        # Training history
        self.training_history = {
            'losses': [],
            'rewards': [],
            'communication_rounds': 0,
            'last_update': 0
        }
        
        # Privacy parameters
        self.privacy_budget = 1.0
        self.noise_scale = 0.1
    
    def add_local_data(self, states: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor):
        """Add local training data."""
        self.local_data.append((states, actions, rewards))
        self.local_data_size += len(states)
    
    def local_training(self, num_epochs: int = 5) -> Dict[str, float]:
        """Perform local training."""
        if not self.local_data:
            return {'loss': 0.0, 'reward': 0.0}
        
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for states, actions, rewards in self.local_data:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                
                # Forward pass
                action_logits = self.local_model(states)
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
                
                # Policy loss
                loss = -(log_probs * rewards.unsqueeze(1)).mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_reward += rewards.mean().item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_reward = total_reward / num_batches if num_batches > 0 else 0.0
        
        # Update training history
        self.training_history['losses'].append(avg_loss)
        self.training_history['rewards'].append(avg_reward)
        
        return {'loss': avg_loss, 'reward': avg_reward}
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get local model parameters."""
        return {name: param.data.clone() for name, param in self.local_model.named_parameters()}
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set local model parameters."""
        for name, param in self.local_model.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name])
    
    def add_differential_privacy_noise(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to parameters."""
        noisy_parameters = {}
        for name, param in parameters.items():
            noise = torch.randn_like(param) * self.noise_scale
            noisy_parameters[name] = param + noise
        
        return noisy_parameters
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'client_id': self.client_id,
            'data_size': self.local_data_size,
            'training_history': self.training_history.copy(),
            'privacy_budget': self.privacy_budget,
            'device': self.device
        }


class FederatedServer:
    """Federated learning server for RL."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Global model
        self.global_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(device)
        
        # Client management
        self.clients = {}
        self.client_weights = {}
        
        # Aggregation parameters
        self.aggregation_method = 'fedavg'  # 'fedavg', 'fedprox', 'feddyn'
        self.mu = 0.01  # FedProx regularization parameter
        
        # Training history
        self.training_history = {
            'global_losses': [],
            'global_rewards': [],
            'communication_rounds': 0,
            'client_participation': []
        }
    
    def add_client(self, client: FederatedClient):
        """Add a client to the federation."""
        self.clients[client.client_id] = client
        self.client_weights[client.client_id] = 1.0  # Equal weighting by default
    
    def remove_client(self, client_id: str):
        """Remove a client from the federation."""
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.client_weights:
            del self.client_weights[client_id]
    
    def aggregate_parameters(self, client_parameters: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters."""
        if not client_parameters:
            return self.get_global_parameters()
        
        # Calculate total weight
        total_weight = sum(self.client_weights[client_id] for client_id in client_parameters.keys())
        
        # Initialize aggregated parameters
        aggregated_params = {}
        for name in self.global_model.state_dict().keys():
            aggregated_params[name] = torch.zeros_like(self.global_model.state_dict()[name])
        
        # Weighted aggregation
        for client_id, params in client_parameters.items():
            weight = self.client_weights[client_id] / total_weight
            for name, param in params.items():
                if name in aggregated_params:
                    aggregated_params[name] += weight * param
        
        return aggregated_params
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters."""
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}
    
    def set_global_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set global model parameters."""
        for name, param in self.global_model.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name])
    
    def communication_round(self, participating_clients: List[str] = None, 
                           local_epochs: int = 5) -> Dict[str, Any]:
        """Perform one communication round."""
        if participating_clients is None:
            participating_clients = list(self.clients.keys())
        
        # Send global model to participating clients
        global_params = self.get_global_parameters()
        for client_id in participating_clients:
            if client_id in self.clients:
                self.clients[client_id].set_model_parameters(global_params)
        
        # Collect local updates
        client_updates = {}
        client_stats = {}
        
        for client_id in participating_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Local training
                local_stats = client.local_training(local_epochs)
                client_stats[client_id] = local_stats
                
                # Get updated parameters
                local_params = client.get_model_parameters()
                
                # Add differential privacy noise if enabled
                if hasattr(client, 'add_differential_privacy_noise'):
                    local_params = client.add_differential_privacy_noise(local_params)
                
                client_updates[client_id] = local_params
        
        # Aggregate updates
        aggregated_params = self.aggregate_parameters(client_updates)
        self.set_global_parameters(aggregated_params)
        
        # Update training history
        self.training_history['communication_rounds'] += 1
        self.training_history['client_participation'].append(participating_clients)
        
        # Calculate global statistics
        global_loss = np.mean([stats['loss'] for stats in client_stats.values()])
        global_reward = np.mean([stats['reward'] for stats in client_stats.values()])
        
        self.training_history['global_losses'].append(global_loss)
        self.training_history['global_rewards'].append(global_reward)
        
        return {
            'global_loss': global_loss,
            'global_reward': global_reward,
            'participating_clients': participating_clients,
            'client_stats': client_stats
        }
    
    def evaluate_global_model(self, test_data: List[Tuple]) -> Dict[str, float]:
        """Evaluate global model on test data."""
        self.global_model.eval()
        total_loss = 0.0
        total_reward = 0.0
        num_episodes = len(test_data)
        
        with torch.no_grad():
            for states, actions, rewards in test_data:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                
                # Forward pass
                action_logits = self.global_model(states)
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
                
                # Loss
                loss = -(log_probs * rewards.unsqueeze(1)).mean()
                
                total_loss += loss.item()
                total_reward += rewards.mean().item()
        
        return {
            'avg_loss': total_loss / num_episodes,
            'avg_reward': total_reward / num_episodes
        }
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'num_clients': len(self.clients),
            'aggregation_method': self.aggregation_method,
            'training_history': self.training_history.copy(),
            'client_weights': self.client_weights.copy()
        }


class FederatedRLAggregator:
    """Federated RL aggregator for managing the federation."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Server
        self.server = FederatedServer(state_dim, action_dim, device)
        
        # Federation parameters
        self.num_communication_rounds = 100
        self.local_epochs = 5
        self.participation_rate = 1.0  # Fraction of clients participating each round
        
        # Evaluation
        self.test_data = []
        self.evaluation_history = []
        
        # Statistics
        self.federation_statistics = {
            'total_rounds': 0,
            'client_statistics': {},
            'convergence_metrics': []
        }
    
    def add_client(self, client_id: str, lr: float = 1e-3) -> FederatedClient:
        """Add a new client to the federation."""
        client = FederatedClient(client_id, self.state_dim, self.action_dim, lr, self.device)
        self.server.add_client(client)
        return client
    
    def add_test_data(self, test_data: List[Tuple]):
        """Add test data for evaluation."""
        self.test_data.extend(test_data)
    
    def train_federation(self, num_rounds: int = None) -> Dict[str, Any]:
        """Train the federation for specified number of rounds."""
        if num_rounds is None:
            num_rounds = self.num_communication_rounds
        
        training_results = {
            'rounds': [],
            'global_losses': [],
            'global_rewards': [],
            'evaluation_results': []
        }
        
        for round_idx in range(num_rounds):
            # Select participating clients
            all_clients = list(self.server.clients.keys())
            num_participating = max(1, int(len(all_clients) * self.participation_rate))
            participating_clients = np.random.choice(
                all_clients, size=num_participating, replace=False
            ).tolist()
            
            # Communication round
            round_results = self.server.communication_round(participating_clients, self.local_epochs)
            
            # Evaluation
            if self.test_data:
                eval_results = self.server.evaluate_global_model(self.test_data)
                self.evaluation_history.append(eval_results)
                training_results['evaluation_results'].append(eval_results)
            
            # Store results
            training_results['rounds'].append(round_idx)
            training_results['global_losses'].append(round_results['global_loss'])
            training_results['global_rewards'].append(round_results['global_reward'])
            
            # Update federation statistics
            self.federation_statistics['total_rounds'] += 1
        
        return training_results
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get statistics for all clients."""
        client_stats = {}
        for client_id, client in self.server.clients.items():
            client_stats[client_id] = client.get_client_statistics()
        
        return client_stats
    
    def get_federation_statistics(self) -> Dict[str, Any]:
        """Get overall federation statistics."""
        stats = {
            'server_statistics': self.server.get_server_statistics(),
            'client_statistics': self.get_client_statistics(),
            'evaluation_history': self.evaluation_history.copy(),
            'federation_parameters': {
                'num_communication_rounds': self.num_communication_rounds,
                'local_epochs': self.local_epochs,
                'participation_rate': self.participation_rate
            }
        }
        
        return stats
    
    def save_federation(self, filepath: str):
        """Save federation state."""
        torch.save({
            'server_state': self.server.global_model.state_dict(),
            'federation_statistics': self.federation_statistics,
            'evaluation_history': self.evaluation_history
        }, filepath)
    
    def load_federation(self, filepath: str):
        """Load federation state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.server.global_model.load_state_dict(checkpoint['server_state'])
        self.federation_statistics = checkpoint['federation_statistics']
        self.evaluation_history = checkpoint['evaluation_history']
