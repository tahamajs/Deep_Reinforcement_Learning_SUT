"""
Quantum-Inspired RL Environment

This module contains environments designed for quantum-inspired reinforcement learning,
including quantum state representations and interference effects.
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
class QuantumState:
    """Quantum state representation."""

    amplitudes: np.ndarray
    phases: np.ndarray
    entanglement: np.ndarray
    coherence: float


class QuantumRLEnvironment(Env):
    """Environment for quantum-inspired RL."""

    def __init__(self, state_dim: int = 4, action_dim: int = 4, num_qubits: int = 2):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_qubits = num_qubits

        # Action space: quantum operations
        self.action_space = spaces.Discrete(action_dim)

        # Observation space: quantum state + classical state
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(state_dim + 2**num_qubits * 2,), dtype=np.float32
        )

        # Quantum state
        self.quantum_state = None
        self.classical_state = np.zeros(state_dim)

        # Environment parameters
        self.decoherence_rate = 0.01
        self.entanglement_strength = 0.5
        self.interference_strength = 0.3

        # Performance tracking
        self.quantum_metrics = {
            "coherence_history": [],
            "entanglement_history": [],
            "interference_events": 0,
            "measurement_events": 0,
        }

        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 200

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Initialize quantum state
        self._initialize_quantum_state()

        # Reset classical state
        self.classical_state = np.random.uniform(-1, 1, self.state_dim)

        # Reset episode tracking
        self.episode_length = 0

        return self.get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Apply quantum operation
        reward = self._apply_quantum_operation(action)

        # Apply decoherence
        self._apply_decoherence()

        # Update classical state based on quantum state
        self._update_classical_state()

        # Update state
        self.episode_length += 1

        # Check termination
        done = self.episode_length >= self.max_episode_length

        # Create info
        info = {
            "quantum_metrics": self.quantum_metrics.copy(),
            "coherence": self.quantum_state.coherence,
            "entanglement": np.mean(self.quantum_state.entanglement),
        }

        return self.get_observation(), reward, done, False, info

    def _initialize_quantum_state(self):
        """Initialize quantum state."""
        num_states = 2**self.num_qubits

        # Initialize amplitudes (normalized)
        amplitudes = np.random.uniform(0, 1, num_states)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        # Initialize phases
        phases = np.random.uniform(0, 2 * np.pi, num_states)

        # Initialize entanglement matrix
        entanglement = np.eye(num_states) * 0.1

        # Initial coherence
        coherence = 1.0

        self.quantum_state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement=entanglement,
            coherence=coherence,
        )

    def _apply_quantum_operation(self, action: int) -> float:
        """Apply quantum operation based on action."""
        reward = 0.0

        if action == 0:  # Pauli-X (bit flip)
            self._apply_pauli_x()
            reward += 0.1
        elif action == 1:  # Pauli-Y
            self._apply_pauli_y()
            reward += 0.1
        elif action == 2:  # Pauli-Z (phase flip)
            self._apply_pauli_z()
            reward += 0.1
        elif action == 3:  # Hadamard (superposition)
            self._apply_hadamard()
            reward += 0.2
        elif action == 4:  # CNOT (entanglement)
            self._apply_cnot()
            reward += 0.3
        elif action == 5:  # Measurement
            measurement_result = self._apply_measurement()
            reward += measurement_result * 0.5
        else:
            # Identity operation
            reward += 0.01

        return reward

    def _apply_pauli_x(self):
        """Apply Pauli-X gate."""
        # Simple bit flip on first qubit
        if self.num_qubits >= 1:
            # Swap amplitudes of |0⟩ and |1⟩ states
            if len(self.quantum_state.amplitudes) >= 2:
                self.quantum_state.amplitudes[0], self.quantum_state.amplitudes[1] = (
                    self.quantum_state.amplitudes[1],
                    self.quantum_state.amplitudes[0],
                )

    def _apply_pauli_y(self):
        """Apply Pauli-Y gate."""
        # Y gate: combination of X and Z
        self._apply_pauli_x()
        self._apply_pauli_z()

    def _apply_pauli_z(self):
        """Apply Pauli-Z gate."""
        # Phase flip on first qubit
        if self.num_qubits >= 1 and len(self.quantum_state.phases) >= 2:
            self.quantum_state.phases[1] += np.pi

    def _apply_hadamard(self):
        """Apply Hadamard gate."""
        # Create superposition
        if self.num_qubits >= 1 and len(self.quantum_state.amplitudes) >= 2:
            # Hadamard transformation
            h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            state_vector = np.array(
                [self.quantum_state.amplitudes[0], self.quantum_state.amplitudes[1]]
            )
            new_state = h_matrix @ state_vector

            self.quantum_state.amplitudes[0] = abs(new_state[0])
            self.quantum_state.amplitudes[1] = abs(new_state[1])
            self.quantum_state.phases[0] = np.angle(new_state[0])
            self.quantum_state.phases[1] = np.angle(new_state[1])

    def _apply_cnot(self):
        """Apply CNOT gate."""
        # Create entanglement between qubits
        if self.num_qubits >= 2:
            # Simple entanglement operation
            num_states = len(self.quantum_state.amplitudes)
            if num_states >= 4:
                # Entangle states |00⟩ and |11⟩
                self.quantum_state.entanglement[0, 3] += self.entanglement_strength
                self.quantum_state.entanglement[3, 0] += self.entanglement_strength

    def _apply_measurement(self) -> float:
        """Apply measurement operation."""
        # Collapse quantum state
        probabilities = self.quantum_state.amplitudes**2

        # Sample measurement outcome
        outcome = np.random.choice(len(probabilities), p=probabilities)

        # Collapse state
        self.quantum_state.amplitudes = np.zeros_like(self.quantum_state.amplitudes)
        self.quantum_state.amplitudes[outcome] = 1.0

        # Update metrics
        self.quantum_metrics["measurement_events"] += 1

        # Return reward based on outcome
        return float(outcome) / len(probabilities)

    def _apply_decoherence(self):
        """Apply decoherence to quantum state."""
        # Reduce coherence
        self.quantum_state.coherence *= 1 - self.decoherence_rate

        # Add random phase noise
        noise = np.random.normal(0, 0.01, len(self.quantum_state.phases))
        self.quantum_state.phases += noise

        # Update metrics
        self.quantum_metrics["coherence_history"].append(self.quantum_state.coherence)

    def _update_classical_state(self):
        """Update classical state based on quantum state."""
        # Use quantum state to influence classical state
        if len(self.quantum_state.amplitudes) >= 2:
            # Use first two amplitudes to update classical state
            self.classical_state[0] = self.quantum_state.amplitudes[0] - 0.5
            self.classical_state[1] = self.quantum_state.amplitudes[1] - 0.5

        # Add quantum interference effects
        if self.quantum_state.coherence > 0.5:
            interference = self.interference_strength * np.sin(
                self.quantum_state.phases[0]
            )
            self.classical_state[2] += interference
            self.quantum_metrics["interference_events"] += 1

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Classical state
        classical_obs = self.classical_state

        # Quantum state (amplitudes and phases)
        quantum_obs = np.concatenate(
            [
                self.quantum_state.amplitudes,
                self.quantum_state.phases / (2 * np.pi),  # Normalize phases
            ]
        )

        obs = np.concatenate([classical_obs, quantum_obs])
        return obs.astype(np.float32)

    def get_quantum_info(self) -> Dict[str, Any]:
        """Get quantum state information."""
        return {
            "amplitudes": self.quantum_state.amplitudes.copy(),
            "phases": self.quantum_state.phases.copy(),
            "coherence": self.quantum_state.coherence,
            "entanglement": np.mean(self.quantum_state.entanglement),
            "num_qubits": self.num_qubits,
        }

    def apply_quantum_noise(self, noise_strength: float = 0.1):
        """Apply quantum noise to the state."""
        # Add noise to amplitudes
        noise = np.random.normal(0, noise_strength, len(self.quantum_state.amplitudes))
        self.quantum_state.amplitudes += noise

        # Renormalize
        norm = np.linalg.norm(self.quantum_state.amplitudes)
        if norm > 0:
            self.quantum_state.amplitudes /= norm

        # Add noise to phases
        phase_noise = np.random.normal(
            0, noise_strength, len(self.quantum_state.phases)
        )
        self.quantum_state.phases += phase_noise

        # Reduce coherence
        self.quantum_state.coherence *= 1 - noise_strength

    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum metrics."""
        return {
            "coherence": self.quantum_state.coherence,
            "entanglement": np.mean(self.quantum_state.entanglement),
            "interference_events": self.quantum_metrics["interference_events"],
            "measurement_events": self.quantum_metrics["measurement_events"],
            "coherence_history": self.quantum_metrics["coherence_history"][-10:],
        }
