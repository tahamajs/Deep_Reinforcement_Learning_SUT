"""
Neuromorphic RL Environment

This module contains environments designed for neuromorphic computing,
including spiking neural networks and event-driven processing.
"""

import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from typing import Dict, List, Any, Optional, Tuple
import random
import time
from dataclasses import dataclass
import math


@dataclass
class SpikeEvent:
    """Spike event representation."""
    timestamp: float
    neuron_id: int
    spike_strength: float
    event_type: str  # "input", "output", "internal"


@dataclass
class NeuronState:
    """State of a spiking neuron."""
    membrane_potential: float
    threshold: float
    refractory_period: float
    last_spike_time: float
    spike_count: int
    adaptation: float


class NeuromorphicRLEnvironment(Env):
    """Environment for neuromorphic RL."""
    
    def __init__(self, num_neurons: int = 100, num_inputs: int = 10, num_outputs: int = 4):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # Action space: output neuron indices
        self.action_space = spaces.Discrete(num_outputs)
        
        # Observation space: membrane potentials + spike rates
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_neurons + num_inputs,), dtype=np.float32
        )
        
        # Neuron states
        self.neurons = {}
        self.input_neurons = {}
        self.output_neurons = {}
        
        # Spike events
        self.spike_events = []
        self.event_queue = []
        
        # Network parameters
        self.membrane_time_constant = 20.0  # ms
        self.threshold = 1.0
        self.refractory_period = 2.0  # ms
        self.adaptation_rate = 0.01
        
        # Synaptic weights
        self.synaptic_weights = np.random.normal(0, 0.1, (num_neurons, num_neurons))
        
        # Performance tracking
        self.neuromorphic_metrics = {
            "total_spikes": 0,
            "spike_rate": 0.0,
            "energy_consumption": 0.0,
            "event_processing_time": 0.0,
        }
        
        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 200
        self.current_time = 0.0
        self.time_step = 0.1  # ms
        
        # Initialize network
        self._initialize_network()
        
    def _initialize_network(self):
        """Initialize the neuromorphic network."""
        # Initialize input neurons
        for i in range(self.num_inputs):
            self.input_neurons[i] = NeuronState(
                membrane_potential=0.0,
                threshold=self.threshold,
                refractory_period=0.0,
                last_spike_time=-1000.0,
                spike_count=0,
                adaptation=0.0,
            )
            
        # Initialize output neurons
        for i in range(self.num_outputs):
            self.output_neurons[i] = NeuronState(
                membrane_potential=0.0,
                threshold=self.threshold,
                refractory_period=0.0,
                last_spike_time=-1000.0,
                spike_count=0,
                adaptation=0.0,
            )
            
        # Initialize internal neurons
        for i in range(self.num_neurons - self.num_inputs - self.num_outputs):
            self.neurons[i] = NeuronState(
                membrane_potential=0.0,
                threshold=self.threshold,
                refractory_period=0.0,
                last_spike_time=-1000.0,
                spike_count=0,
                adaptation=0.0,
            )
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset neuron states
        for neuron in self.input_neurons.values():
            neuron.membrane_potential = 0.0
            neuron.refractory_period = 0.0
            neuron.last_spike_time = -1000.0
            neuron.spike_count = 0
            neuron.adaptation = 0.0
            
        for neuron in self.output_neurons.values():
            neuron.membrane_potential = 0.0
            neuron.refractory_period = 0.0
            neuron.last_spike_time = -1000.0
            neuron.spike_count = 0
            neuron.adaptation = 0.0
            
        for neuron in self.neurons.values():
            neuron.membrane_potential = 0.0
            neuron.refractory_period = 0.0
            neuron.last_spike_time = -1000.0
            neuron.spike_count = 0
            neuron.adaptation = 0.0
            
        # Reset spike events
        self.spike_events.clear()
        self.event_queue.clear()
        
        # Reset episode tracking
        self.episode_length = 0
        self.current_time = 0.0
        
        return self.get_observation(), {}
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Process input spikes
        self._process_input_spikes()
        
        # Process network dynamics
        self._process_network_dynamics()
        
        # Process output action
        reward = self._process_output_action(action)
        
        # Update state
        self.episode_length += 1
        self.current_time += self.time_step
        
        # Check termination
        done = self.episode_length >= self.max_episode_length
        
        # Create info
        info = {
            "neuromorphic_metrics": self.neuromorphic_metrics.copy(),
            "spike_rate": self._calculate_spike_rate(),
            "energy_consumption": self._calculate_energy_consumption(),
        }
        
        return self.get_observation(), reward, done, False, info
        
    def _process_input_spikes(self):
        """Process input spikes."""
        # Generate random input spikes
        for i, neuron in self.input_neurons.items():
            if random.random() < 0.1:  # 10% chance of spike
                self._generate_spike(i, "input", 1.0)
                
    def _process_network_dynamics(self):
        """Process network dynamics."""
        # Update membrane potentials
        for i, neuron in self.neurons.items():
            self._update_membrane_potential(neuron)
            
        for i, neuron in self.output_neurons.items():
            self._update_membrane_potential(neuron)
            
        # Process synaptic transmission
        self._process_synaptic_transmission()
        
        # Check for spikes
        self._check_for_spikes()
        
    def _update_membrane_potential(self, neuron: NeuronState):
        """Update membrane potential of a neuron."""
        # Leaky integrate-and-fire dynamics
        if neuron.refractory_period > 0:
            neuron.refractory_period -= self.time_step
            neuron.membrane_potential = 0.0
        else:
            # Decay membrane potential
            decay_factor = math.exp(-self.time_step / self.membrane_time_constant)
            neuron.membrane_potential *= decay_factor
            
            # Add adaptation
            neuron.membrane_potential -= neuron.adaptation
            
            # Decay adaptation
            neuron.adaptation *= (1 - self.adaptation_rate)
            
    def _process_synaptic_transmission(self):
        """Process synaptic transmission."""
        # Process spike events
        for event in self.spike_events:
            if event.timestamp <= self.current_time:
                self._transmit_spike(event)
                
        # Remove processed events
        self.spike_events = [e for e in self.spike_events if e.timestamp > self.current_time]
        
    def _transmit_spike(self, event: SpikeEvent):
        """Transmit a spike through synapses."""
        if event.event_type == "input":
            # Input spike affects internal neurons
            for i, neuron in self.neurons.items():
                weight = self.synaptic_weights[i, event.neuron_id]
                neuron.membrane_potential += weight * event.spike_strength
                
        elif event.event_type == "internal":
            # Internal spike affects other neurons
            for i, neuron in self.neurons.items():
                if i != event.neuron_id:
                    weight = self.synaptic_weights[i, event.neuron_id]
                    neuron.membrane_potential += weight * event.spike_strength
                    
            # Also affect output neurons
            for i, neuron in self.output_neurons.items():
                weight = self.synaptic_weights[i + len(self.neurons), event.neuron_id]
                neuron.membrane_potential += weight * event.spike_strength
                
    def _check_for_spikes(self):
        """Check for spikes and generate events."""
        # Check internal neurons
        for i, neuron in self.neurons.items():
            if neuron.membrane_potential >= neuron.threshold and neuron.refractory_period <= 0:
                self._generate_spike(i, "internal", 1.0)
                neuron.membrane_potential = 0.0
                neuron.refractory_period = self.refractory_period
                neuron.last_spike_time = self.current_time
                neuron.spike_count += 1
                neuron.adaptation += 0.1
                
        # Check output neurons
        for i, neuron in self.output_neurons.items():
            if neuron.membrane_potential >= neuron.threshold and neuron.refractory_period <= 0:
                self._generate_spike(i, "output", 1.0)
                neuron.membrane_potential = 0.0
                neuron.refractory_period = self.refractory_period
                neuron.last_spike_time = self.current_time
                neuron.spike_count += 1
                neuron.adaptation += 0.1
                
    def _generate_spike(self, neuron_id: int, event_type: str, strength: float):
        """Generate a spike event."""
        event = SpikeEvent(
            timestamp=self.current_time + 0.1,  # Small delay
            neuron_id=neuron_id,
            spike_strength=strength,
            event_type=event_type,
        )
        self.spike_events.append(event)
        self.neuromorphic_metrics["total_spikes"] += 1
        
    def _process_output_action(self, action: int) -> float:
        """Process output action."""
        if action < len(self.output_neurons):
            # Stimulate output neuron
            neuron = self.output_neurons[action]
            neuron.membrane_potential += 0.5
            
            # Calculate reward based on spike activity
            reward = 0.0
            if neuron.spike_count > 0:
                reward = 1.0 / (1.0 + neuron.spike_count)  # Reward for controlled spiking
                
            return reward
            
        return 0.0
        
    def _calculate_spike_rate(self) -> float:
        """Calculate overall spike rate."""
        total_spikes = sum(neuron.spike_count for neuron in self.neurons.values())
        total_spikes += sum(neuron.spike_count for neuron in self.output_neurons.values())
        total_spikes += sum(neuron.spike_count for neuron in self.input_neurons.values())
        
        return total_spikes / max(1, self.current_time)
        
    def _calculate_energy_consumption(self) -> float:
        """Calculate energy consumption."""
        # Energy is proportional to spike count
        total_spikes = self.neuromorphic_metrics["total_spikes"]
        energy_per_spike = 0.1  # pJ per spike
        return total_spikes * energy_per_spike
        
    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Membrane potentials
        membrane_potentials = []
        
        # Input neuron potentials
        for neuron in self.input_neurons.values():
            membrane_potentials.append(neuron.membrane_potential)
            
        # Internal neuron potentials
        for neuron in self.neurons.values():
            membrane_potentials.append(neuron.membrane_potential)
            
        # Output neuron potentials
        for neuron in self.output_neurons.values():
            membrane_potentials.append(neuron.membrane_potential)
            
        # Normalize to [0, 1]
        membrane_potentials = np.array(membrane_potentials)
        membrane_potentials = np.clip(membrane_potentials, 0, 1)
        
        return membrane_potentials.astype(np.float32)
        
    def get_neuromorphic_info(self) -> Dict[str, Any]:
        """Get neuromorphic network information."""
        return {
            "num_neurons": self.num_neurons,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "spike_rate": self._calculate_spike_rate(),
            "energy_consumption": self._calculate_energy_consumption(),
            "total_spikes": self.neuromorphic_metrics["total_spikes"],
            "current_time": self.current_time,
        }
        
    def apply_plasticity(self, learning_rate: float = 0.01):
        """Apply synaptic plasticity (STDP)."""
        # Simple STDP rule
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    # Get spike times
                    pre_spike_time = self.neurons.get(j, NeuronState(0, 0, 0, -1000, 0, 0)).last_spike_time
                    post_spike_time = self.neurons.get(i, NeuronState(0, 0, 0, -1000, 0, 0)).last_spike_time
                    
                    # Calculate time difference
                    if pre_spike_time > 0 and post_spike_time > 0:
                        time_diff = post_spike_time - pre_spike_time
                        
                        # STDP rule
                        if time_diff > 0:  # Post after pre (LTP)
                            weight_change = learning_rate * math.exp(-time_diff / 10.0)
                        else:  # Pre after post (LTD)
                            weight_change = -learning_rate * math.exp(time_diff / 10.0)
                            
                        # Update weight
                        self.synaptic_weights[i, j] += weight_change
                        self.synaptic_weights[i, j] = np.clip(self.synaptic_weights[i, j], -1, 1)
                        
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network performance metrics."""
        return {
            "spike_rate": self._calculate_spike_rate(),
            "energy_consumption": self._calculate_energy_consumption(),
            "total_spikes": self.neuromorphic_metrics["total_spikes"],
            "avg_membrane_potential": np.mean([n.membrane_potential for n in self.neurons.values()]),
            "avg_adaptation": np.mean([n.adaptation for n in self.neurons.values()]),
            "synaptic_weight_stats": {
                "mean": np.mean(self.synaptic_weights),
                "std": np.std(self.synaptic_weights),
                "min": np.min(self.synaptic_weights),
                "max": np.max(self.synaptic_weights),
            },
        }
