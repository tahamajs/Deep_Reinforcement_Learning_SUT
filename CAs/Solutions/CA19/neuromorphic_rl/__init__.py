"""
Neuromorphic Reinforcement Learning Module

This module implements brain-inspired reinforcement learning using spiking neural networks,
event-driven processing, and biologically plausible learning rules like STDP.

Key Components:
- SpikingNeuron: Leaky integrate-and-fire neuron model
- STDPSynapse: Spike-timing-dependent plasticity learning
- SpikingNetwork: Event-driven neural network architecture
- NeuromorphicActorCritic: Brain-inspired RL agent with dopamine modulation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import random
from collections import deque
import warnings

warnings.filterwarnings("ignore")


class SpikingNeuron:
    """
    Leaky Integrate-and-Fire Neuron Model

    Biologically inspired neuron that integrates inputs over time and fires
    when membrane potential exceeds a threshold.
    """

    def __init__(
        self,
        membrane_time_constant: float = 0.01,
        threshold: float = 1.0,
        reset_potential: float = 0.0,
        refractory_period: float = 0.002,
    ):
        self.tau = membrane_time_constant  # Membrane time constant
        self.threshold = threshold
        self.reset = reset_potential
        self.refractory = refractory_period

        # Neuron state
        self.v = reset_potential  # Membrane potential
        self.last_spike_time = -np.inf
        self.refractory_timer = 0.0

    def step(self, input_current: float, dt: float = 0.001) -> Tuple[bool, float]:
        """
        Update neuron state for one time step

        Args:
            input_current: Input current to the neuron
            dt: Time step size

        Returns:
            Tuple of (spike_fired, membrane_potential)
        """
        # Update refractory timer
        if self.refractory_timer > 0:
            self.refractory_timer -= dt
            return False, self.v

        # Leaky integration
        dv = (-self.v + input_current) / self.tau * dt
        self.v += dv

        # Check for spike
        if self.v >= self.threshold:
            # Fire spike
            self.v = self.reset
            self.refractory_timer = self.refractory
            return True, self.threshold
        else:
            return False, self.v

    def reset_state(self):
        """Reset neuron to initial state"""
        self.v = self.reset
        self.last_spike_time = -np.inf
        self.refractory_timer = 0.0


class STDPSynapse:
    """
    Spike-Timing-Dependent Plasticity Synapse

    Learning rule that modifies synaptic strength based on the relative timing
    of pre- and post-synaptic spikes, implementing Hebbian learning.
    """

    def __init__(
        self,
        initial_weight: float = 0.5,
        a_plus: float = 0.05,
        a_minus: float = 0.03,
        tau_plus: float = 0.02,
        tau_minus: float = 0.02,
    ):
        self.weight = initial_weight
        self.a_plus = a_plus  # LTP amplitude
        self.a_minus = a_minus  # LTD amplitude
        self.tau_plus = tau_plus  # LTP time constant
        self.tau_minus = tau_minus  # LTD time constant

        # STDP state
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf

    def pre_spike(self, current_time: float):
        """Handle pre-synaptic spike"""
        self.last_pre_spike = current_time

        # Apply LTD if post-synaptic spike occurred recently
        if self.last_post_spike > 0:
            time_diff = current_time - self.last_post_spike
            if time_diff > 0:  # Pre after post
                delta_w = -self.a_minus * np.exp(-time_diff / self.tau_minus)
                self.weight = np.clip(self.weight + delta_w, 0.0, 1.0)

    def post_spike(self, current_time: float):
        """Handle post-synaptic spike"""
        self.last_post_spike = current_time

        # Apply LTP if pre-synaptic spike occurred recently
        if self.last_pre_spike > 0:
            time_diff = current_time - self.last_pre_spike
            if time_diff > 0:  # Post after pre
                delta_w = self.a_plus * np.exp(-time_diff / self.tau_plus)
                self.weight = np.clip(self.weight + delta_w, 0.0, 1.0)

    def get_weight(self) -> float:
        """Get current synaptic weight"""
        return self.weight

    def set_weight(self, weight: float):
        """Set synaptic weight"""
        self.weight = np.clip(weight, 0.0, 1.0)


class SpikingNetwork:
    """
    Event-Driven Spiking Neural Network

    Network of spiking neurons connected by plastic synapses,
    implementing event-driven processing for energy-efficient computation.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        dt: float = 0.001,
        connectivity: float = 0.1,
    ):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dt = dt

        # Create neurons
        self.input_neurons = [SpikingNeuron() for _ in range(n_input)]
        self.hidden_neurons = [SpikingNeuron() for _ in range(n_hidden)]
        self.output_neurons = [SpikingNeuron() for _ in range(n_output)]

        # Create synapses
        self.input_hidden_synapses = [
            [STDPSynapse() for _ in range(n_hidden)] for _ in range(n_input)
        ]
        self.hidden_output_synapses = [
            [STDPSynapse() for _ in range(n_output)] for _ in range(n_hidden)
        ]

        # Random initialization of connectivity
        for i in range(n_input):
            for j in range(n_hidden):
                if np.random.random() < connectivity:
                    self.input_hidden_synapses[i][j].set_weight(np.random.random())

        for i in range(n_hidden):
            for j in range(n_output):
                if np.random.random() < connectivity:
                    self.hidden_output_synapses[i][j].set_weight(np.random.random())

    def reset(self):
        """Reset all neurons to initial state"""
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            neuron.reset_state()

    def forward(
        self, input_spikes: np.ndarray, n_steps: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the spiking network

        Args:
            input_spikes: Binary input spike train (n_input,)
            n_steps: Number of time steps to simulate

        Returns:
            Tuple of (output_spike_trains, hidden_spike_trains)
        """
        # Initialize spike trains
        output_spikes = np.zeros((n_steps, self.n_output))
        hidden_spikes = np.zeros((n_steps, self.n_hidden))

        current_time = 0.0

        for t in range(n_steps):
            # Input layer
            input_currents = np.zeros(self.n_hidden)
            for i, spike in enumerate(input_spikes):
                if spike > 0.5:  # Input spike
                    for j in range(self.n_hidden):
                        input_currents[j] += self.input_hidden_synapses[i][
                            j
                        ].get_weight()

            # Hidden layer
            hidden_currents = np.zeros(self.n_output)
            for j in range(self.n_hidden):
                spike_fired, _ = self.hidden_neurons[j].step(input_currents[j], self.dt)
                hidden_spikes[t, j] = 1.0 if spike_fired else 0.0

                if spike_fired:
                    self.hidden_neurons[j].last_spike_time = current_time
                    # Update STDP for input synapses
                    for i in range(self.n_input):
                        if input_spikes[i] > 0.5:
                            self.input_hidden_synapses[i][j].post_spike(current_time)

                # Accumulate currents to output layer
                for k in range(self.n_output):
                    hidden_currents[k] += (
                        self.hidden_output_synapses[j][k].get_weight()
                        * hidden_spikes[t, j]
                    )

            # Output layer
            for k in range(self.n_output):
                spike_fired, _ = self.output_neurons[k].step(
                    hidden_currents[k], self.dt
                )
                output_spikes[t, k] = 1.0 if spike_fired else 0.0

                if spike_fired:
                    self.output_neurons[k].last_spike_time = current_time
                    # Update STDP for hidden synapses
                    for j in range(self.n_hidden):
                        if hidden_spikes[t, j] > 0.5:
                            self.hidden_output_synapses[j][k].post_spike(current_time)

            current_time += self.dt

        return output_spikes, hidden_spikes

    def get_output_rates(self) -> np.ndarray:
        """Get average firing rates of output neurons"""
        rates = []
        for k in range(self.n_output):
            # Count spikes over recent history (simplified)
            spike_count = sum(
                1 for t in range(len(self.output_neurons[k].last_spike_time > 0))
            )
            rate = spike_count / max(1, len(self.output_neurons[k].last_spike_time > 0))
            rates.append(rate)
        return np.array(rates)

    def get_synaptic_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current synaptic weight matrices"""
        input_hidden_weights = np.array(
            [
                [
                    self.input_hidden_synapses[i][j].get_weight()
                    for j in range(self.n_hidden)
                ]
                for i in range(self.n_input)
            ]
        )
        hidden_output_weights = np.array(
            [
                [
                    self.hidden_output_synapses[j][k].get_weight()
                    for k in range(self.n_output)
                ]
                for j in range(self.n_hidden)
            ]
        )
        return input_hidden_weights, hidden_output_weights


class NeuromorphicActorCritic:
    """
    Neuromorphic Actor-Critic Agent

    Brain-inspired RL agent using spiking neural networks with dopamine-modulated
    learning and event-driven processing for energy-efficient reinforcement learning.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        dt: float = 0.01,
        dopamine_baseline: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.dopamine_baseline = dopamine_baseline

        # Spiking neural networks
        self.actor_network = SpikingNetwork(obs_dim, hidden_dim, action_dim, dt)
        self.critic_network = SpikingNetwork(
            obs_dim, hidden_dim, 1, dt
        )  # Single value output

        # Dopamine system for reward modulation
        self.dopamine_level = dopamine_baseline
        self.dopamine_tau = 0.1  # Dopamine time constant

        # Learning parameters
        self.td_error = 0.0
        self.learning_rate = 0.01

        # Performance tracking
        self.episode_rewards = []
        self.td_errors = []
        self.dopamine_levels = []
        self.firing_rates = []

    def reset_networks(self):
        """Reset spiking networks to initial state"""
        self.actor_network.reset()
        self.critic_network.reset()
        self.dopamine_level = self.dopamine_baseline

    def select_action(self, observation: np.ndarray) -> Tuple[int, Dict]:
        """
        Select action using neuromorphic processing

        Args:
            observation: Current observation

        Returns:
            Tuple of (action, action_info)
        """
        # Convert observation to spike encoding
        spike_input = self._encode_observation(observation)

        # Forward pass through actor network
        output_spikes, hidden_spikes = self.actor_network.forward(
            spike_input, n_steps=10
        )

        # Decode action from output firing rates
        firing_rates = self.actor_network.get_output_rates()
        action = np.argmax(firing_rates)

        # Get value estimate from critic
        critic_spikes, _ = self.critic_network.forward(spike_input, n_steps=10)
        value_estimate = self.critic_network.get_output_rates()[0]

        action_info = {
            "firing_rates": firing_rates,
            "value_estimate": value_estimate,
            "dopamine_level": self.dopamine_level,
            "td_error": self.td_error,
            "method": "neuromorphic",
        }

        return action, action_info

    def _encode_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Encode continuous observation into spike train

        Uses rate coding: higher values -> higher firing rates
        """
        # Normalize observation
        obs_norm = (observation - np.min(observation)) / (
            np.max(observation) - np.min(observation) + 1e-6
        )

        # Rate encoding: probability of spike proportional to value
        spike_probs = np.clip(obs_norm, 0, 1)
        spike_input = (np.random.random(len(spike_probs)) < spike_probs).astype(float)

        return spike_input

    def update_dopamine(self, reward: float, next_value: float, current_value: float):
        """
        Update dopamine level based on reward prediction error

        Args:
            reward: Immediate reward
            next_value: Next state value estimate
            current_value: Current state value estimate
        """
        # TD error calculation
        self.td_error = reward + 0.99 * next_value - current_value

        # Dopamine release proportional to TD error
        dopamine_release = self.dopamine_baseline + self.td_error * 0.1

        # Update dopamine level with temporal dynamics
        self.dopamine_level += (
            (dopamine_release - self.dopamine_level) / self.dopamine_tau * self.dt
        )

        # Clip dopamine to reasonable range
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)

    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> Dict:
        """
        Learn from experience using neuromorphic plasticity

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Episode termination flag

        Returns:
            Learning metrics
        """
        # Get current value estimate
        current_spike_input = self._encode_observation(observation)
        _, _ = self.critic_network.forward(current_spike_input, n_steps=5)
        current_value = self.critic_network.get_output_rates()[0]

        # Get next value estimate
        if not done:
            next_spike_input = self._encode_observation(next_observation)
            _, _ = self.critic_network.forward(next_spike_input, n_steps=5)
            next_value = self.critic_network.get_output_rates()[0]
        else:
            next_value = 0.0

        # Update dopamine system
        self.update_dopamine(reward, next_value, current_value)

        # Store metrics
        self.td_errors.append(self.td_error)
        self.dopamine_levels.append(self.dopamine_level)

        # Get actor firing rates for learning
        firing_rates = self.actor_network.get_output_rates()
        self.firing_rates.append(np.mean(firing_rates))

        # Neuromorphic learning is primarily through STDP and dopamine modulation
        # The spiking networks learn online through spike timing
        # Here we could implement additional supervised learning if needed

        learning_info = {
            "td_error": self.td_error,
            "dopamine_level": self.dopamine_level,
            "avg_firing_rate": np.mean(firing_rates),
            "value_estimate": current_value,
        }

        return learning_info

    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        return {
            "avg_td_error": np.mean(self.td_errors[-100:]) if self.td_errors else 0.0,
            "avg_dopamine": (
                np.mean(self.dopamine_levels[-100:]) if self.dopamine_levels else 0.0
            ),
            "avg_firing_rate": (
                np.mean(self.firing_rates[-100:]) if self.firing_rates else 0.0
            ),
            "current_dopamine": self.dopamine_level,
            "network_complexity": {
                "input_neurons": self.obs_dim,
                "hidden_neurons": self.hidden_dim,
                "output_neurons": self.action_dim,
            },
        }
