"""
Advanced Reinforcement Learning Algorithms
الگوریتم‌های پیشرفته یادگیری تقویتی

This module contains cutting-edge RL algorithms including:
- Hierarchical RL with Options
- Meta-Learning (MAML)
- Causal RL
- Quantum-Inspired RL
- Neurosymbolic RL
- Federated RL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np
import copy
from typing import Dict, List, Tuple, Any, Optional
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HierarchicalRLAgent:
    """Hierarchical RL Agent with Options Framework."""
    
    def __init__(self, state_dim, action_dim, num_options=5, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        
        # Meta-policy (option selection)
        self.meta_policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_options),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Option policies
        self.option_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(num_options)
        ]).to(device)
        
        # Termination functions
        self.termination_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(num_options)
        ]).to(device)
        
        # Value functions
        self.meta_value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        self.option_values = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_options)
        ]).to(device)
        
        self.optimizer = optim.Adam(
            list(self.meta_policy.parameters()) +
            list(self.option_policies.parameters()) +
            list(self.termination_functions.parameters()) +
            list(self.meta_value.parameters()) +
            list(self.option_values.parameters()),
            lr=lr
        )
        
        self.gamma = 0.99
        self.beta = 0.01  # Termination penalty
        
    def select_option(self, state):
        """Select option using meta-policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            option_probs = self.meta_policy(state_tensor)
            option_dist = Categorical(option_probs)
            option = option_dist.sample()
            return option.item()
    
    def get_action(self, state, current_option):
        """Get action from current option policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.option_policies[current_option](state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item()
    
    def should_terminate(self, state, current_option):
        """Check if current option should terminate."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            termination_prob = self.termination_functions[current_option](state_tensor)
            return torch.rand(1).item() < termination_prob.item()
    
    def update(self, trajectories):
        """Update hierarchical agent."""
        if not trajectories:
            return None
            
        total_loss = 0
        meta_losses = []
        option_losses = []
        
        for trajectory in trajectories:
            states, actions, rewards, options, terminations = zip(*trajectory)
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            options = torch.LongTensor(options).to(device)
            terminations = torch.BoolTensor(terminations).to(device)
            
            # Compute returns
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(device)
            
            # Meta-policy loss
            meta_log_probs = torch.log(self.meta_policy(states) + 1e-8)
            meta_log_probs = meta_log_probs.gather(1, options.unsqueeze(1)).squeeze()
            meta_loss = -(meta_log_probs * returns).mean()
            
            # Option policy losses
            option_loss = 0
            for i, option in enumerate(options):
                option_log_probs = torch.log(self.option_policies[option](states[i:i+1]) + 1e-8)
                option_log_probs = option_log_probs.gather(1, actions[i:i+1].unsqueeze(1)).squeeze()
                option_loss += -(option_log_probs * returns[i]).mean()
            
            # Termination losses
            termination_loss = 0
            for i, option in enumerate(options):
                if terminations[i]:
                    termination_prob = self.termination_functions[option](states[i:i+1])
                    termination_loss += -torch.log(termination_prob + 1e-8).mean()
            
            total_loss += meta_loss + option_loss + termination_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.meta_policy.parameters()) +
            list(self.option_policies.parameters()) +
            list(self.termination_functions.parameters()) +
            list(self.meta_value.parameters()) +
            list(self.option_values.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "meta_loss": meta_loss.item(),
            "option_loss": option_loss.item(),
            "termination_loss": termination_loss.item()
        }


class MetaLearningAgent:
    """Model-Agnostic Meta-Learning (MAML) for RL."""
    
    def __init__(self, state_dim, action_dim, inner_lr=0.01, meta_lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        
        # Meta-network
        self.meta_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        self.meta_optimizer = optim.Adam(self.meta_network.parameters(), lr=meta_lr)
        
        # Task-specific networks
        self.task_networks = {}
        
    def adapt_to_task(self, task_id, support_data, num_steps=5):
        """Adapt to a specific task using support data."""
        if task_id not in self.task_networks:
            self.task_networks[task_id] = copy.deepcopy(self.meta_network)
        
        task_network = self.task_networks[task_id]
        task_optimizer = optim.SGD(task_network.parameters(), lr=self.inner_lr)
        
        # Inner loop adaptation
        for step in range(num_steps):
            task_optimizer.zero_grad()
            
            states, actions, rewards = support_data
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            
            action_probs = task_network(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            loss = -log_probs.mean()
            
            loss.backward()
            task_optimizer.step()
        
        return task_network
    
    def meta_update(self, tasks_data):
        """Meta-update using multiple tasks."""
        meta_loss = 0
        
        for task_id, (support_data, query_data) in tasks_data.items():
            # Adapt to task
            adapted_network = self.adapt_to_task(task_id, support_data)
            
            # Evaluate on query data
            query_states, query_actions, query_rewards = query_data
            query_states = torch.FloatTensor(query_states).to(device)
            query_actions = torch.LongTensor(query_actions).to(device)
            
            query_action_probs = adapted_network(query_states)
            query_log_probs = torch.log(query_action_probs.gather(1, query_actions.unsqueeze(1)).squeeze() + 1e-8)
            
            meta_loss += -query_log_probs.mean()
        
        # Meta-gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class CausalRLAgent:
    """Causal Reinforcement Learning Agent."""
    
    def __init__(self, state_dim, action_dim, causal_graph=None, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_graph = causal_graph or self._create_default_causal_graph()
        
        # Causal policy network
        self.causal_policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Intervention networks
        self.intervention_networks = nn.ModuleDict({
            node: nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for node in self.causal_graph.keys()
        }).to(device)
        
        # Counterfactual network
        self.counterfactual_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.causal_policy.parameters()) +
            list(self.intervention_networks.parameters()) +
            list(self.counterfactual_network.parameters()),
            lr=lr
        )
        
    def _create_default_causal_graph(self):
        """Create default causal graph."""
        return {
            'position': ['reward'],
            'velocity': ['position', 'reward'],
            'action': ['velocity', 'reward']
        }
    
    def intervene(self, state, intervention_node, intervention_value):
        """Perform causal intervention."""
        intervened_state = state.copy()
        intervened_state[intervention_node] = intervention_value
        
        # Propagate intervention through causal graph
        for node, parents in self.causal_graph.items():
            if intervention_node in parents:
                parent_values = [intervened_state[parent] for parent in parents]
                intervention_input = torch.FloatTensor(parent_values).unsqueeze(0).to(device)
                new_value = self.intervention_networks[node](intervention_input)
                intervened_state[node] = new_value.item()
        
        return intervened_state
    
    def counterfactual_reasoning(self, state, action, alternative_action):
        """Perform counterfactual reasoning."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor([action]).unsqueeze(0).to(device)
        alt_action_tensor = torch.FloatTensor([alternative_action]).unsqueeze(0).to(device)
        
        # Original outcome
        original_input = torch.cat([state_tensor, action_tensor], dim=1)
        original_outcome = self.counterfactual_network(original_input)
        
        # Counterfactual outcome
        counterfactual_input = torch.cat([state_tensor, alt_action_tensor], dim=1)
        counterfactual_outcome = self.counterfactual_network(counterfactual_input)
        
        return original_outcome, counterfactual_outcome
    
    def get_action(self, state):
        """Get action using causal policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.causal_policy(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item()
    
    def update(self, trajectories):
        """Update causal agent."""
        if not trajectories:
            return None
        
        total_loss = 0
        causal_losses = []
        intervention_losses = []
        counterfactual_losses = []
        
        for trajectory in trajectories:
            states, actions, rewards, interventions, counterfactuals = zip(*trajectory)
            
            # Causal policy loss
            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.LongTensor(actions).to(device)
            
            action_probs = self.causal_policy(states_tensor)
            log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)
            causal_loss = -log_probs.mean()
            
            # Intervention loss
            intervention_loss = 0
            for i, (state, intervention) in enumerate(zip(states, interventions)):
                if intervention:
                    node, value = intervention
                    intervened_state = self.intervene(state, node, value)
                    # Compute loss based on intervention effectiveness
                    intervention_loss += F.mse_loss(
                        torch.FloatTensor(intervened_state).to(device),
                        torch.FloatTensor(state).to(device)
                    )
            
            # Counterfactual loss
            counterfactual_loss = 0
            for i, (state, action, counterfactual) in enumerate(zip(states, actions, counterfactuals)):
                if counterfactual:
                    alt_action = counterfactual
                    original_outcome, counterfactual_outcome = self.counterfactual_reasoning(
                        state, action, alt_action
                    )
                    counterfactual_loss += F.mse_loss(original_outcome, counterfactual_outcome)
            
            total_loss += causal_loss + intervention_loss + counterfactual_loss
            causal_losses.append(causal_loss.item())
            intervention_losses.append(intervention_loss.item())
            counterfactual_losses.append(counterfactual_loss.item())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.causal_policy.parameters()) +
            list(self.intervention_networks.parameters()) +
            list(self.counterfactual_network.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "causal_loss": np.mean(causal_losses),
            "intervention_loss": np.mean(intervention_losses),
            "counterfactual_loss": np.mean(counterfactual_losses)
        }


class QuantumInspiredRLAgent:
    """Quantum-Inspired Reinforcement Learning Agent."""
    
    def __init__(self, state_dim, action_dim, num_qubits=4, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_qubits = num_qubits
        self.state_dimension = 2 ** num_qubits
        
        # Quantum state representation
        self.quantum_state = nn.Parameter(torch.randn(self.state_dimension, dtype=torch.complex64))
        
        # Quantum gates (parameterized)
        self.rotation_gates = nn.Parameter(torch.randn(num_qubits, 3))  # X, Y, Z rotations
        self.entanglement_gates = nn.Parameter(torch.randn(num_qubits - 1, 2))  # CNOT parameters
        
        # Measurement operators
        self.measurement_operators = nn.ModuleList([
            nn.Linear(self.state_dimension, action_dim) for _ in range(action_dim)
        ]).to(device)
        
        # Classical policy network
        self.classical_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        self.optimizer = optim.Adam(
            [self.quantum_state, self.rotation_gates, self.entanglement_gates] +
            list(self.measurement_operators.parameters()) +
            list(self.classical_policy.parameters()),
            lr=lr
        )
        
    def apply_quantum_gate(self, state, gate_type, qubit_idx, angle):
        """Apply quantum gate to state."""
        if gate_type == 'X':
            gate = torch.tensor([[torch.cos(angle/2), -1j*torch.sin(angle/2)],
                               [-1j*torch.sin(angle/2), torch.cos(angle/2)]], dtype=torch.complex64)
        elif gate_type == 'Y':
            gate = torch.tensor([[torch.cos(angle/2), -torch.sin(angle/2)],
                               [torch.sin(angle/2), torch.cos(angle/2)]], dtype=torch.complex64)
        elif gate_type == 'Z':
            gate = torch.tensor([[torch.exp(-1j*angle/2), 0],
                               [0, torch.exp(1j*angle/2)]], dtype=torch.complex64)
        
        # Apply gate to specific qubit
        return state  # Simplified implementation
    
    def entangle_qubits(self, state, qubit1, qubit2):
        """Apply entanglement between qubits."""
        # CNOT gate implementation (simplified)
        return state
    
    def quantum_measurement(self, state):
        """Perform quantum measurement."""
        probabilities = torch.abs(state) ** 2
        return probabilities
    
    def get_action(self, state):
        """Get action using quantum-inspired policy."""
        with torch.no_grad():
            # Classical component
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            classical_probs = self.classical_policy(state_tensor)
            
            # Quantum component
            quantum_probs = self.quantum_measurement(self.quantum_state)
            quantum_probs = quantum_probs[:self.action_dim]  # Truncate to action space
            
            # Combine classical and quantum
            combined_probs = 0.7 * classical_probs + 0.3 * quantum_probs
            combined_probs = combined_probs / combined_probs.sum()
            
            action_dist = Categorical(combined_probs)
            action = action_dist.sample()
            return action.item()
    
    def update(self, trajectories):
        """Update quantum-inspired agent."""
        if not trajectories:
            return None
        
        total_loss = 0
        quantum_losses = []
        classical_losses = []
        
        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)
            
            # Classical loss
            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.LongTensor(actions).to(device)
            
            classical_probs = self.classical_policy(states_tensor)
            classical_log_probs = torch.log(classical_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)
            classical_loss = -classical_log_probs.mean()
            
            # Quantum loss (based on measurement probabilities)
            quantum_probs = self.quantum_measurement(self.quantum_state)
            quantum_probs = quantum_probs[:self.action_dim]
            quantum_log_probs = torch.log(quantum_probs.gather(0, actions_tensor) + 1e-8)
            quantum_loss = -quantum_log_probs.mean()
            
            total_loss += classical_loss + quantum_loss
            classical_losses.append(classical_loss.item())
            quantum_losses.append(quantum_loss.item())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "classical_loss": np.mean(classical_losses),
            "quantum_loss": np.mean(quantum_losses)
        }


class NeurosymbolicRLAgent:
    """Neuro-Symbolic Reinforcement Learning Agent."""
    
    def __init__(self, state_dim, action_dim, symbolic_dim=32, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.symbolic_dim = symbolic_dim
        
        # Neural component
        self.neural_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, symbolic_dim)
        ).to(device)
        
        # Symbolic reasoning component
        self.symbolic_reasoner = nn.Sequential(
            nn.Linear(symbolic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, symbolic_dim)
        ).to(device)
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(symbolic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Symbolic rules (learnable)
        self.symbolic_rules = nn.Parameter(torch.randn(symbolic_dim, symbolic_dim))
        
        self.optimizer = optim.Adam(
            list(self.neural_encoder.parameters()) +
            list(self.symbolic_reasoner.parameters()) +
            list(self.policy_network.parameters()) +
            [self.symbolic_rules],
            lr=lr
        )
        
    def symbolic_reasoning(self, symbolic_state):
        """Perform symbolic reasoning."""
        # Apply symbolic rules
        reasoned_state = torch.matmul(symbolic_state, self.symbolic_rules)
        
        # Apply symbolic reasoner
        reasoned_state = self.symbolic_reasoner(reasoned_state)
        
        return reasoned_state
    
    def get_action(self, state):
        """Get action using neuro-symbolic policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Neural encoding
            symbolic_state = self.neural_encoder(state_tensor)
            
            # Symbolic reasoning
            reasoned_state = self.symbolic_reasoning(symbolic_state)
            
            # Policy decision
            action_probs = self.policy_network(reasoned_state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            
            return action.item()
    
    def update(self, trajectories):
        """Update neuro-symbolic agent."""
        if not trajectories:
            return None
        
        total_loss = 0
        neural_losses = []
        symbolic_losses = []
        
        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)
            
            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.LongTensor(actions).to(device)
            
            # Neural encoding
            symbolic_states = self.neural_encoder(states_tensor)
            
            # Symbolic reasoning
            reasoned_states = self.symbolic_reasoning(symbolic_states)
            
            # Policy loss
            action_probs = self.policy_network(reasoned_states)
            log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)
            policy_loss = -log_probs.mean()
            
            # Symbolic consistency loss
            symbolic_loss = F.mse_loss(symbolic_states, reasoned_states)
            
            total_loss += policy_loss + symbolic_loss
            neural_losses.append(policy_loss.item())
            symbolic_losses.append(symbolic_loss.item())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "neural_loss": np.mean(neural_losses),
            "symbolic_loss": np.mean(symbolic_losses)
        }


class FederatedRLAgent:
    """Federated Reinforcement Learning Agent."""
    
    def __init__(self, state_dim, action_dim, num_clients=5, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_clients = num_clients
        
        # Global model
        self.global_policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        # Client models
        self.client_policies = nn.ModuleList([
            copy.deepcopy(self.global_policy) for _ in range(num_clients)
        ]).to(device)
        
        # Client optimizers
        self.client_optimizers = [
            optim.Adam(client.parameters(), lr=lr) for client in self.client_policies
        ]
        
        self.global_optimizer = optim.Adam(self.global_policy.parameters(), lr=lr)
        
        # Federated learning parameters
        self.federation_rounds = 0
        self.client_data_sizes = [0] * num_clients
        
    def local_update(self, client_id, trajectories):
        """Perform local update on client."""
        if not trajectories:
            return None
        
        client_policy = self.client_policies[client_id]
        client_optimizer = self.client_optimizers[client_id]
        
        total_loss = 0
        for trajectory in trajectories:
            states, actions, rewards = zip(*trajectory)
            
            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.LongTensor(actions).to(device)
            
            action_probs = client_policy(states_tensor)
            log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)
            loss = -log_probs.mean()
            total_loss += loss
        
        # Local update
        client_optimizer.zero_grad()
        total_loss.backward()
        client_optimizer.step()
        
        self.client_data_sizes[client_id] = len(trajectories)
        
        return total_loss.item()
    
    def federated_averaging(self):
        """Perform federated averaging."""
        # Collect client parameters
        client_params = []
        for client_policy in self.client_policies:
            client_params.append(list(client_policy.parameters()))
        
        # Compute weighted average
        global_params = list(self.global_policy.parameters())
        
        for i, global_param in enumerate(global_params):
            weighted_sum = torch.zeros_like(global_param)
            total_weight = 0
            
            for j, client_param in enumerate(client_params):
                weight = self.client_data_sizes[j]
                weighted_sum += weight * client_param[i]
                total_weight += weight
            
            if total_weight > 0:
                global_param.data = weighted_sum / total_weight
        
        # Update client models with global model
        for client_policy in self.client_policies:
            client_policy.load_state_dict(self.global_policy.state_dict())
        
        self.federation_rounds += 1
        
        return self.federation_rounds
    
    def get_action(self, state, client_id=None):
        """Get action from global or client policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            if client_id is not None:
                action_probs = self.client_policies[client_id](state_tensor)
            else:
                action_probs = self.global_policy(state_tensor)
            
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item()
    
    def update(self, client_trajectories):
        """Update federated agent."""
        if not client_trajectories:
            return None
        
        client_losses = []
        
        # Local updates
        for client_id, trajectories in enumerate(client_trajectories):
            if trajectories:
                loss = self.local_update(client_id, trajectories)
                client_losses.append(loss)
        
        # Federated averaging
        federation_round = self.federated_averaging()
        
        return {
            "client_losses": client_losses,
            "federation_round": federation_round,
            "avg_client_loss": np.mean(client_losses) if client_losses else 0
        }
