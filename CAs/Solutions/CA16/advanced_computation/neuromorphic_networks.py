"""
Neuromorphic Networks for Advanced Computation

This module provides neuromorphic computing approaches for reinforcement learning,
including spiking neural networks and neuromorphic processors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import math


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Basic spiking neuron that integrates inputs and fires when threshold is reached.
    """

    def __init__(self, tau: float = 20.0, threshold: float = 1.0, reset: float = 0.0):
        super().__init__()
        self.tau = tau  # Membrane time constant
        self.threshold = threshold
        self.reset = reset

        self.v = None  # Membrane potential
        self.spike = None  # Spike output

    def forward(
        self, x: torch.Tensor, dt: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LIF neuron.

        Args:
            x: Input current
            dt: Time step

        Returns:
            Spike output and membrane potential
        """
        if self.v is None:
            self.v = torch.zeros_like(x)

        dv = (-self.v + x) * (dt / self.tau)
        self.v = self.v + dv

        self.spike = (self.v >= self.threshold).float()

        self.v = self.v * (1 - self.spike) + self.reset * self.spike

        return self.spike, self.v

    def reset_state(self):
        """Reset neuron state."""
        self.v = None
        self.spike = None


class AdaptiveLIFNeuron(LIFNeuron):
    """
    Adaptive LIF neuron with spike-frequency adaptation.

    Includes adaptation current that reduces excitability after spikes.
    """

    def __init__(
        self,
        tau: float = 20.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        tau_adapt: float = 100.0,
        beta: float = 0.1,
    ):
        super().__init__(tau, threshold, reset)
        self.tau_adapt = tau_adapt  # Adaptation time constant
        self.beta = beta  # Adaptation strength

        self.adapt = None

    def forward(
        self, x: torch.Tensor, dt: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptation.

        Args:
            x: Input current
            dt: Time step

        Returns:
            Spike output and membrane potential
        """
        if self.v is None:
            self.v = torch.zeros_like(x)
        if self.adapt is None:
            self.adapt = torch.zeros_like(x)

        dadapt = -self.adapt * (dt / self.tau_adapt) + self.beta * self.spike
        self.adapt = self.adapt + dadapt

        effective_input = x - self.adapt

        dv = (-self.v + effective_input) * (dt / self.tau)
        self.v = self.v + dv

        self.spike = (self.v >= self.threshold).float()

        self.v = self.v * (1 - self.spike) + self.reset * self.spike

        return self.spike, self.v

    def reset_state(self):
        """Reset neuron state."""
        super().reset_state()
        self.adapt = None


class SpikingNeuralNetwork(nn.Module):
    """
    Spiking Neural Network for temporal processing.

    Processes input sequences using spiking neurons with temporal dynamics.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        neuron_type: str = "lif",
        dt: float = 1.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dt = dt

        self.input_encoder = nn.Linear(input_size, hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        prev_size = hidden_sizes[0]

        for hidden_size in hidden_sizes[1:]:
            layer = nn.Linear(prev_size, hidden_size)
            self.hidden_layers.append(layer)
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, output_size)

        self.neurons = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            if neuron_type == "lif":
                neuron = LIFNeuron()
            elif neuron_type == "adaptive_lif":
                neuron = AdaptiveLIFNeuron()
            else:
                raise ValueError(f"Unknown neuron type: {neuron_type}")
            self.neurons.append(neuron)

        if neuron_type == "lif":
            self.output_neuron = LIFNeuron()
        else:
            self.output_neuron = AdaptiveLIFNeuron()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through SNN.

        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            Output spikes and spike history
        """
        batch_size, seq_len, _ = x.shape
        spike_history = []

        for neuron in self.neurons:
            neuron.reset_state()
        self.output_neuron.reset_state()

        for t in range(seq_len):
            current_input = x[:, t, :]

            hidden_input = self.input_encoder(current_input)
            hidden_input = torch.relu(hidden_input)

            for i, (layer, neuron) in enumerate(zip(self.hidden_layers, self.neurons)):
                if i == 0:
                    spike, _ = neuron(hidden_input, self.dt)
                else:
                    spike, _ = neuron(hidden_input, self.dt)

                hidden_input = layer(spike)
                spike_history.append(spike.detach())

            output_input = self.output_layer(hidden_input)
            output_spike, _ = self.output_neuron(output_input, self.dt)
            spike_history.append(output_spike.detach())

        return output_spike, spike_history

    def get_spike_rate(self, spike_history: List[torch.Tensor]) -> torch.Tensor:
        """Compute average spike rate from history."""
        if not spike_history:
            return torch.tensor(0.0)

        total_spikes = sum(torch.sum(spikes) for spikes in spike_history)
        total_neurons = sum(spikes.numel() for spikes in spike_history)
        total_time = len(spike_history)

        return total_spikes / (total_neurons * total_time)


class NeuromorphicNetwork(nn.Module):
    """
    Neuromorphic Network for RL.

    Combines spiking neural networks with reinforcement learning
    for energy-efficient temporal processing.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [64, 32],
        sequence_length: int = 10,
        neuron_type: str = "adaptive_lif",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length

        self.state_buffer = deque(maxlen=sequence_length)

        self.policy_net = SpikingNeuralNetwork(
            input_size=state_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            neuron_type=neuron_type,
        )

        self.value_net = nn.Sequential(
            nn.Linear(state_dim * sequence_length, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(
                hidden_sizes[0],
                hidden_sizes[1] if len(hidden_sizes) > 1 else hidden_sizes[0],
            ),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1] if len(hidden_sizes) > 1 else hidden_sizes[0], 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through neuromorphic network.

        Args:
            state: Current state

        Returns:
            Action logits and value estimate
        """
        self.state_buffer.append(state.clone())

        while len(self.state_buffer) < self.sequence_length:
            self.state_buffer.append(torch.zeros_like(state))

        sequence = torch.stack(
            list(self.state_buffer), dim=1
        )  # (batch, seq_len, state_dim)

        action_logits, spike_history = self.policy_net(sequence)

        flat_sequence = sequence.view(sequence.size(0), -1)
        value = self.value_net(flat_sequence)

        return action_logits, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from neuromorphic policy.

        Args:
            state: Current state
            deterministic: Whether to return deterministic action

        Returns:
            Action and log probability
        """
        logits, _ = self.forward(state)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        log_prob = torch.log_softmax(logits, dim=-1)[action]

        return action, log_prob

    def get_spike_statistics(self) -> Dict[str, float]:
        """Get spiking statistics."""
        return {"avg_spike_rate": 0.0, "total_spikes": 0, "network_activity": 0.0}


class NeuromorphicProcessor:
    """
    Neuromorphic Processor for efficient RL computation.

    Simulates neuromorphic hardware constraints and capabilities.
    """

    def __init__(
        self,
        max_neurons: int = 1000,
        max_synapses: int = 10000,
        energy_budget: float = 100.0,
    ):
        self.max_neurons = max_neurons
        self.max_synapses = max_synapses
        self.energy_budget = energy_budget

        self.neuron_count = 0
        self.synapse_count = 0
        self.energy_used = 0.0

        self.computation_time = 0.0
        self.power_consumption = 0.0

    def allocate_neurons(self, count: int) -> bool:
        """
        Allocate neurons on neuromorphic hardware.

        Args:
            count: Number of neurons to allocate

        Returns:
            True if allocation successful
        """
        if self.neuron_count + count <= self.max_neurons:
            self.neuron_count += count
            return True
        return False

    def allocate_synapses(self, count: int) -> bool:
        """
        Allocate synapses on neuromorphic hardware.

        Args:
            count: Number of synapses to allocate

        Returns:
            True if allocation successful
        """
        if self.synapse_count + count <= self.max_synapses:
            self.synapse_count += count
            return True
        return False

    def process_spikes(
        self, spike_train: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Process spike train through synaptic connections.

        Args:
            spike_train: Input spike train
            weights: Synaptic weight matrix

        Returns:
            Output currents
        """
        current = torch.matmul(spike_train.float(), weights)

        spike_count = torch.sum(spike_train).item()
        energy_cost = spike_count * 0.01  # Energy per spike

        self.energy_used += energy_cost
        self.computation_time += 0.001  # Processing time

        return current

    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        return {
            "neuron_utilization": self.neuron_count / self.max_neurons,
            "synapse_utilization": self.synapse_count / self.max_synapses,
            "energy_utilization": self.energy_used / self.energy_budget,
            "computation_time": self.computation_time,
        }

    def reset(self):
        """Reset processor state."""
        self.neuron_count = 0
        self.synapse_count = 0
        self.energy_used = 0.0
        self.computation_time = 0.0
        self.power_consumption = 0.0

    def optimize_for_energy(self, network: nn.Module) -> nn.Module:
        """
        Optimize network for energy efficiency.

        Args:
            network: Neural network to optimize

        Returns:
            Energy-optimized network
        """

        optimized_network = copy.deepcopy(network)

        for module in optimized_network.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weights = module.weight.data
                    threshold = torch.quantile(torch.abs(weights), 0.1)
                    mask = torch.abs(weights) > threshold
                    module.weight.data *= mask

        return optimized_network
