"""
Advanced Neuromorphic Reinforcement Learning Agent

This module implements sophisticated neuromorphic RL algorithms including:
- Multi-scale Spiking Neural Networks
- Hierarchical Temporal Processing
- Adaptive Plasticity Rules
- Neuromodulation and Attention Mechanisms
- Energy-efficient Learning
- Biological Plausibility Constraints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import warnings
warnings.filterwarnings("ignore")

class AdaptiveSTDPSynapse:
    """
    Advanced STDP synapse with adaptive learning rates and metaplasticity
    """
    
    def __init__(self, initial_weight: float = 0.5, a_plus: float = 0.05, 
                 a_minus: float = 0.03, tau_plus: float = 20.0, tau_minus: float = 20.0):
        self.weight = initial_weight
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        
        # Adaptive learning rates
        self.adaptive_lr_plus = a_plus
        self.adaptive_lr_minus = a_minus
        
        # Metaplasticity parameters
        self.metaplasticity_threshold = 0.1
        self.metaplasticity_factor = 0.9
        
        # Spike history
        self.pre_spike_times = deque(maxlen=100)
        self.post_spike_times = deque(maxlen=100)
        
        # Weight bounds
        self.min_weight = -2.0
        self.max_weight = 2.0
        
    def pre_spike(self, time: float):
        """Process presynaptic spike"""
        self.pre_spike_times.append(time)
        
        # LTP for post spikes that occurred before this pre spike
        for post_time in self.post_spike_times:
            dt = time - post_time
            if dt > 0:
                ltp = self.adaptive_lr_plus * np.exp(-dt / self.tau_plus)
                self.weight += ltp
                
                # Adaptive learning rate update
                if self.weight > self.metaplasticity_threshold:
                    self.adaptive_lr_plus *= self.metaplasticity_factor
        
        # Clamp weight
        self.weight = np.clip(self.weight, self.min_weight, self.max_weight)
    
    def post_spike(self, time: float):
        """Process postsynaptic spike"""
        self.post_spike_times.append(time)
        
        # LTD for pre spikes that occurred before this post spike
        for pre_time in self.pre_spike_times:
            dt = time - pre_time
            if dt > 0:
                ltd = self.adaptive_lr_minus * np.exp(-dt / self.tau_minus)
                self.weight -= ltd
                
                # Adaptive learning rate update
                if self.weight < -self.metaplasticity_threshold:
                    self.adaptive_lr_minus *= self.metaplasticity_factor
        
        # Clamp weight
        self.weight = np.clip(self.weight, self.min_weight, self.max_weight)
    
    def get_weight(self) -> float:
        """Get current synaptic weight"""
        return self.weight


class NeuromodulatedNeuron:
    """
    Advanced neuron model with neuromodulation and attention mechanisms
    """
    
    def __init__(self, neuron_id: int, membrane_time_constant: float = 0.01,
                 threshold: float = 1.0, reset_potential: float = 0.0,
                 refractory_period: float = 0.002):
        self.neuron_id = neuron_id
        self.tau = membrane_time_constant
        self.threshold = threshold
        self.reset = reset_potential
        self.refractory = refractory_period
        
        # Membrane potential and state
        self.v = reset_potential
        self.last_spike_time = -np.inf
        self.refractory_timer = 0.0
        
        # Neuromodulation
        self.dopamine_level = 0.0
        self.acetylcholine_level = 0.0
        self.norepinephrine_level = 0.0
        
        # Attention mechanisms
        self.attention_weight = 1.0
        self.attention_decay = 0.95
        
        # Adaptation
        self.adaptation_current = 0.0
        self.adaptation_time_constant = 0.1
        self.adaptation_increment = 0.1
        
        # Spike history
        self.spike_history = deque(maxlen=1000)
        
    def step(self, input_current: float, dt: float = 0.001) -> Tuple[bool, float]:
        """Update neuron state with neuromodulation"""
        # Update refractory timer
        if self.refractory_timer > 0:
            self.refractory_timer -= dt
            self.v = self.reset
            return False, self.v
        
        # Neuromodulation effects
        neuromodulation_factor = (1.0 + 0.1 * self.dopamine_level + 
                                 0.05 * self.acetylcholine_level + 
                                 0.03 * self.norepinephrine_level)
        
        # Attention modulation
        attention_factor = self.attention_weight
        
        # Membrane potential update
        dv_dt = (-self.v + input_current * neuromodulation_factor * attention_factor - 
                 self.adaptation_current) / self.tau
        self.v += dv_dt * dt
        
        # Adaptation current update
        self.adaptation_current += (-self.adaptation_current / self.adaptation_time_constant + 
                                   self.adaptation_increment * (self.v > 0)) * dt
        
        # Check for spike
        spiked = False
        if self.v >= self.threshold:
            spiked = True
            self.v = self.reset
            self.refractory_timer = self.refractory
            self.last_spike_time = 0.0  # Relative time
            
            # Update spike history
            self.spike_history.append(0.0)
            
            # Increase adaptation
            self.adaptation_current += self.adaptation_increment
        
        # Update attention weight
        self.attention_weight = (self.attention_weight * self.attention_decay + 
                                (1 - self.attention_decay) * (1.0 if spiked else 0.5))
        
        # Update spike times
        for i in range(len(self.spike_history)):
            self.spike_history[i] += dt
        
        return spiked, self.v
    
    def update_neuromodulation(self, dopamine: float, acetylcholine: float, 
                              norepinephrine: float):
        """Update neuromodulator levels"""
        self.dopamine_level = dopamine
        self.acetylcholine_level = acetylcholine
        self.norepinephrine_level = norepinephrine


class HierarchicalSpikingNetwork:
    """
    Multi-scale hierarchical spiking neural network
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Create hierarchical layers
        self.layers = []
        self.synapses = []
        
        # Input layer
        input_neurons = [NeuromodulatedNeuron(i) for i in range(input_dim)]
        self.layers.append(input_neurons)
        
        # Hidden layers
        for layer_idx, hidden_dim in enumerate(hidden_dims):
            hidden_neurons = [NeuromodulatedNeuron(i) for i in range(hidden_dim)]
            self.layers.append(hidden_neurons)
            
            # Create synapses between layers
            prev_layer_size = len(self.layers[-2])
            layer_synapses = []
            
            for i in range(hidden_dim):
                neuron_synapses = []
                for j in range(prev_layer_size):
                    synapse = AdaptiveSTDPSynapse(
                        initial_weight=np.random.normal(0, 0.1),
                        a_plus=0.05 + 0.01 * layer_idx,  # Layer-dependent learning
                        a_minus=0.03 + 0.01 * layer_idx
                    )
                    neuron_synapses.append(synapse)
                layer_synapses.append(neuron_synapses)
            
            self.synapses.append(layer_synapses)
        
        # Output layer
        output_neurons = [NeuromodulatedNeuron(i) for i in range(output_dim)]
        self.layers.append(output_neurons)
        
        # Output layer synapses
        last_hidden_size = len(self.layers[-2])
        output_synapses = []
        for i in range(output_dim):
            neuron_synapses = []
            for j in range(last_hidden_size):
                synapse = AdaptiveSTDPSynapse(
                    initial_weight=np.random.normal(0, 0.1),
                    a_plus=0.08,  # Higher learning rate for output
                    a_minus=0.05
                )
                neuron_synapses.append(synapse)
            output_synapses.append(neuron_synapses)
        
        self.synapses.append(output_synapses)
        
        # Network state
        self.time_step = 0.0
        self.dt = 0.001
        
        # Neuromodulation system
        self.global_dopamine = 0.0
        self.global_acetylcholine = 0.0
        self.global_norepinephrine = 0.0
        
        # Performance tracking
        self.spike_rates = []
        self.synaptic_weights = []
        
    def forward(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Forward pass through hierarchical network"""
        # Encode input as spike rates
        input_rates = np.tanh(input_data) * 100  # Convert to Hz
        
        # Initialize network state
        outputs = np.zeros(self.output_dim)
        network_metrics = {}
        
        # Simulate for multiple time steps
        simulation_time = 0.1  # 100ms simulation
        num_steps = int(simulation_time / self.dt)
        
        spike_counts = np.zeros(self.output_dim)
        
        for step in range(num_steps):
            # Update input layer
            for i, neuron in enumerate(self.layers[0]):
                input_current = input_rates[i] * self.dt
                spiked, _ = neuron.step(input_current, self.dt)
                
                if spiked:
                    # Propagate spike through synapses
                    self._propagate_spike(0, i, self.time_step)
            
            # Update hidden layers
            for layer_idx in range(1, len(self.layers) - 1):
                for i, neuron in enumerate(self.layers[layer_idx]):
                    # Calculate input from previous layer
                    input_current = 0.0
                    for j, synapse in enumerate(self.synapses[layer_idx - 1][i]):
                        input_current += synapse.get_weight() * (1.0 if self.layers[layer_idx - 1][j].last_spike_time == 0.0 else 0.0)
                    
                    spiked, _ = neuron.step(input_current, self.dt)
                    
                    if spiked:
                        self._propagate_spike(layer_idx, i, self.time_step)
            
            # Update output layer
            for i, neuron in enumerate(self.layers[-1]):
                # Calculate input from last hidden layer
                input_current = 0.0
                for j, synapse in enumerate(self.synapses[-1][i]):
                    input_current += synapse.get_weight() * (1.0 if self.layers[-2][j].last_spike_time == 0.0 else 0.0)
                
                spiked, _ = neuron.step(input_current, self.dt)
                
                if spiked:
                    spike_counts[i] += 1
            
            # Update neuromodulation
            self._update_global_neuromodulation()
            
            self.time_step += self.dt
        
        # Calculate output spike rates
        for i in range(self.output_dim):
            outputs[i] = spike_counts[i] / simulation_time
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics()
        
        return outputs, network_metrics
    
    def _propagate_spike(self, layer_idx: int, neuron_idx: int, time: float):
        """Propagate spike through synapses"""
        if layer_idx < len(self.synapses):
            for synapse in self.synapses[layer_idx][neuron_idx]:
                synapse.pre_spike(time)
    
    def _update_global_neuromodulation(self):
        """Update global neuromodulator levels"""
        # Calculate network activity
        total_activity = 0.0
        for layer in self.layers:
            for neuron in layer:
                if len(neuron.spike_history) > 0:
                    total_activity += 1.0
        
        # Update neuromodulators based on activity
        self.global_dopamine = np.tanh(total_activity / 100.0)
        self.global_acetylcholine = 0.5 + 0.3 * np.sin(self.time_step * 10)
        self.global_norepinephrine = 0.3 + 0.2 * np.cos(self.time_step * 5)
        
        # Update all neurons
        for layer in self.layers:
            for neuron in layer:
                neuron.update_neuromodulation(
                    self.global_dopamine,
                    self.global_acetylcholine,
                    self.global_norepinephrine
                )
    
    def _calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate network performance metrics"""
        # Average spike rate
        total_spikes = 0
        total_neurons = 0
        for layer in self.layers:
            for neuron in layer:
                total_spikes += len(neuron.spike_history)
                total_neurons += 1
        
        avg_spike_rate = total_spikes / (total_neurons * 0.1)  # 100ms simulation
        
        # Synaptic weight statistics
        all_weights = []
        for layer_synapses in self.synapses:
            for neuron_synapses in layer_synapses:
                for synapse in neuron_synapses:
                    all_weights.append(synapse.get_weight())
        
        weight_mean = np.mean(all_weights) if all_weights else 0.0
        weight_std = np.std(all_weights) if all_weights else 0.0
        
        # Energy efficiency (spikes per weight update)
        weight_updates = sum(len(layer_synapses) * len(neuron_synapses) 
                           for layer_synapses in self.synapses 
                           for neuron_synapses in layer_synapses)
        energy_efficiency = total_spikes / (weight_updates + 1e-8)
        
        return {
            'avg_spike_rate': avg_spike_rate,
            'weight_mean': weight_mean,
            'weight_std': weight_std,
            'energy_efficiency': energy_efficiency,
            'global_dopamine': self.global_dopamine,
            'global_acetylcholine': self.global_acetylcholine,
            'global_norepinephrine': self.global_norepinephrine
        }
    
    def reset_network(self):
        """Reset network state"""
        for layer in self.layers:
            for neuron in layer:
                neuron.v = neuron.reset
                neuron.last_spike_time = -np.inf
                neuron.refractory_timer = 0.0
                neuron.spike_history.clear()
                neuron.adaptation_current = 0.0
        
        self.time_step = 0.0
        self.global_dopamine = 0.0
        self.global_acetylcholine = 0.0
        self.global_norepinephrine = 0.0


class AdvancedNeuromorphicAgent:
    """
    Advanced neuromorphic RL agent with hierarchical processing
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 32],
                 learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Hierarchical spiking network
        self.spiking_network = HierarchicalSpikingNetwork(
            state_dim, hidden_dims, action_dim
        )
        
        # Classical networks for value estimation and policy refinement
        self.value_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Optimizers
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Performance tracking
        self.episode_rewards = []
        self.network_metrics_history = []
        
    def select_action(self, state: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """Select action using neuromorphic processing"""
        # Get spiking network output
        spike_outputs, network_metrics = self.spiking_network.forward(state)
        
        # Combine with classical policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            classical_probs = self.policy_network(state_tensor).squeeze().numpy()
        
        # Weight combination (spiking network provides exploration, classical provides exploitation)
        combined_probs = 0.7 * classical_probs + 0.3 * spike_outputs
        
        # Normalize probabilities
        combined_probs = combined_probs / (np.sum(combined_probs) + 1e-8)
        
        # Sample action
        action = np.random.choice(self.action_dim, p=combined_probs)
        
        # Store network metrics
        self.network_metrics_history.append(network_metrics)
        
        action_info = {
            'spike_outputs': spike_outputs.tolist(),
            'classical_probs': classical_probs.tolist(),
            'combined_probs': combined_probs.tolist(),
            **network_metrics
        }
        
        return action, action_info
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience for learning"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.memory.append(experience)
    
    def learn(self, batch_size: int = 32):
        """Learn from stored experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(self.memory, batch_size, replace=False)
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        
        # Update value network
        self.value_optimizer.zero_grad()
        
        # Current values
        state_action_pairs = torch.cat([states, torch.zeros(states.size(0), self.action_dim).scatter_(1, actions.unsqueeze(1), 1)], dim=1)
        current_values = self.value_network(state_action_pairs).squeeze()
        
        # Target values
        with torch.no_grad():
            # Get next actions from policy
            next_action_probs = self.policy_network(next_states)
            next_actions = torch.multinomial(next_action_probs, 1).squeeze()
            next_state_action_pairs = torch.cat([next_states, torch.zeros(next_states.size(0), self.action_dim).scatter_(1, next_actions.unsqueeze(1), 1)], dim=1)
            next_values = self.value_network(next_state_action_pairs).squeeze()
            
            target_values = rewards + 0.99 * next_values * (~dones)
        
        value_loss = nn.MSELoss()(current_values, target_values)
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        
        action_probs = self.policy_network(states)
        log_probs = torch.log(action_probs + 1e-8)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calculate advantages
        with torch.no_grad():
            advantages = target_values - current_values
        
        policy_loss = -(selected_log_probs * advantages).mean()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        if not self.network_metrics_history:
            return {}
        
        recent_metrics = self.network_metrics_history[-100:]
        
        avg_spike_rate = np.mean([m['avg_spike_rate'] for m in recent_metrics])
        avg_weight_mean = np.mean([m['weight_mean'] for m in recent_metrics])
        avg_weight_std = np.mean([m['weight_std'] for m in recent_metrics])
        avg_energy_efficiency = np.mean([m['energy_efficiency'] for m in recent_metrics])
        avg_dopamine = np.mean([m['global_dopamine'] for m in recent_metrics])
        avg_acetylcholine = np.mean([m['global_acetylcholine'] for m in recent_metrics])
        avg_norepinephrine = np.mean([m['global_norepinephrine'] for m in recent_metrics])
        
        return {
            'avg_spike_rate': avg_spike_rate,
            'avg_weight_mean': avg_weight_mean,
            'avg_weight_std': avg_weight_std,
            'avg_energy_efficiency': avg_energy_efficiency,
            'avg_dopamine': avg_dopamine,
            'avg_acetylcholine': avg_acetylcholine,
            'avg_norepinephrine': avg_norepinephrine,
            'memory_size': len(self.memory),
            'total_episodes': len(self.episode_rewards)
        }
    
    def reset_network(self):
        """Reset spiking network state"""
        self.spiking_network.reset_network()

