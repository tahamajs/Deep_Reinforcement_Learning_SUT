"""
Quantum Reinforcement Learning

This module provides quantum computing approaches for reinforcement learning,
including quantum agents, circuits, and quantum environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import math


class QuantumGate:
    """Base class for quantum gates."""

    def __init__(self, name: str, matrix: torch.Tensor):
        self.name = name
        self.matrix = matrix

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Apply gate to quantum state."""
        return torch.matmul(self.matrix, state)


class PauliX(QuantumGate):
    """Pauli-X (NOT) gate."""

    def __init__(self):
        matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        super().__init__("X", matrix)


class PauliY(QuantumGate):
    """Pauli-Y gate."""

    def __init__(self):
        matrix = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        super().__init__("Y", matrix)


class PauliZ(QuantumGate):
    """Pauli-Z gate."""

    def __init__(self):
        matrix = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        super().__init__("Z", matrix)


class HadamardGate(QuantumGate):
    """Hadamard gate."""

    def __init__(self):
        matrix = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)
        super().__init__("H", matrix)


class RotationGate(QuantumGate):
    """Rotation gate around specified axis."""

    def __init__(self, axis: str, angle: float):
        if axis == "x":
            matrix = torch.tensor(
                [
                    [torch.cos(angle / 2), -1j * torch.sin(angle / 2)],
                    [-1j * torch.sin(angle / 2), torch.cos(angle / 2)],
                ],
                dtype=torch.complex64,
            )
        elif axis == "y":
            matrix = torch.tensor(
                [
                    [torch.cos(angle / 2), -torch.sin(angle / 2)],
                    [torch.sin(angle / 2), torch.cos(angle / 2)],
                ],
                dtype=torch.complex64,
            )
        elif axis == "z":
            matrix = torch.tensor(
                [[torch.exp(-1j * angle / 2), 0], [0, torch.exp(1j * angle / 2)]],
                dtype=torch.complex64,
            )
        else:
            raise ValueError(f"Unknown rotation axis: {axis}")

        super().__init__(f"R{axis}({angle:.3f})", matrix)


class QuantumCircuit:
    """
    Quantum circuit for variational quantum algorithms.

    Represents a parameterized quantum circuit that can be used
    as part of a hybrid quantum-classical neural network.
    """

    def __init__(self, num_qubits: int, depth: int = 3):
        self.num_qubits = num_qubits
        self.depth = depth

        # Initialize qubits in |0⟩ state
        self.initial_state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        self.initial_state[0] = 1.0

        # Variational parameters (angles for rotation gates)
        self.parameters = nn.Parameter(torch.randn(depth * num_qubits * 3) * 0.1)

        # Common gates
        self.hadamard = HadamardGate()
        self.pauli_x = PauliX()
        self.pauli_y = PauliY()
        self.pauli_z = PauliZ()

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Execute the quantum circuit.

        Args:
            x: Classical input to encode (optional)

        Returns:
            Final quantum state
        """
        state = self.initial_state.clone()

        param_idx = 0

        for layer in range(self.depth):
            # Variational layer
            for qubit in range(self.num_qubits):
                # Rotation gates with learnable parameters
                rx_angle = self.parameters[param_idx]
                ry_angle = self.parameters[param_idx + 1]
                rz_angle = self.parameters[param_idx + 2]

                # Apply rotations
                rx_gate = RotationGate("x", rx_angle)
                ry_gate = RotationGate("y", ry_angle)
                rz_gate = RotationGate("z", rz_angle)

                # Apply to specific qubit (tensor product structure)
                state = self._apply_single_qubit_gate(state, rx_gate, qubit)
                state = self._apply_single_qubit_gate(state, ry_gate, qubit)
                state = self._apply_single_qubit_gate(state, rz_gate, qubit)

                param_idx += 3

            # Entangling layer (CNOT gates)
            for qubit in range(self.num_qubits - 1):
                state = self._apply_cnot(state, qubit, qubit + 1)

        # If classical input provided, encode it
        if x is not None:
            state = self._encode_classical_input(state, x)

        return state

    def _apply_single_qubit_gate(
        self, state: torch.Tensor, gate: QuantumGate, qubit: int
    ) -> torch.Tensor:
        """Apply single-qubit gate to specific qubit."""
        num_states = 2**self.num_qubits

        # Build full operator using tensor products
        op = torch.tensor([[1.0]], dtype=torch.complex64)

        for q in range(self.num_qubits):
            if q == qubit:
                op = torch.kron(op, gate.matrix)
            else:
                identity = torch.eye(2, dtype=torch.complex64)
                op = torch.kron(op, identity)

        return torch.matmul(op, state)

    def _apply_cnot(
        self, state: torch.Tensor, control: int, target: int
    ) -> torch.Tensor:
        """Apply CNOT gate between control and target qubits."""
        num_states = 2**self.num_qubits

        # CNOT matrix construction
        cnot_matrix = torch.zeros((num_states, num_states), dtype=torch.complex64)

        for i in range(num_states):
            # Convert to binary representation
            bits = [(i >> q) & 1 for q in range(self.num_qubits)]

            if bits[control] == 1:  # Control qubit is |1⟩
                # Flip target qubit
                bits[target] = 1 - bits[target]

            # Convert back to index
            j = sum(bit << q for q, bit in enumerate(bits))
            cnot_matrix[j, i] = 1.0

        return torch.matmul(cnot_matrix, state)

    def _encode_classical_input(
        self, state: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Encode classical input into quantum state."""
        # Simple amplitude encoding
        if len(x.shape) > 1:
            x = x.flatten()

        # Normalize input
        x_norm = torch.norm(x)
        if x_norm > 0:
            x = x / x_norm

        # Encode into amplitudes (simplified)
        num_amplitudes = min(len(state), len(x))
        state[:num_amplitudes] = state[:num_amplitudes] + x[:num_amplitudes].to(
            torch.complex64
        )

        # Renormalize
        state = state / torch.norm(state)

        return state

    def get_expectation_value(self, observable: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation value of an observable.

        Args:
            observable: Observable matrix

        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        state = self.forward()
        expectation = torch.conj(state).T @ observable @ state
        return expectation.real

    def measure(self, shots: int = 1000) -> Dict[str, int]:
        """
        Measure the quantum state.

        Args:
            shots: Number of measurement shots

        Returns:
            Measurement outcomes
        """
        state = self.forward()
        probabilities = torch.abs(state) ** 2

        # Sample from probability distribution
        outcomes = torch.multinomial(probabilities.real, shots, replacement=True)

        # Count occurrences
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome.item(), f"0{self.num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts


class QuantumRLAgent(nn.Module):
    """
    Quantum Reinforcement Learning Agent.

    Uses quantum circuits as part of the policy network in a hybrid
    quantum-classical reinforcement learning approach.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_qubits: int = 4,
        circuit_depth: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_qubits = num_qubits

        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits * 2),  # Amplitude and phase encoding
        )

        # Quantum circuit
        self.quantum_circuit = QuantumCircuit(num_qubits, circuit_depth)

        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(2**num_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value network (classical)
        self.value_net = nn.Sequential(
            nn.Linear(2**num_qubits, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through quantum-classical network.

        Args:
            state: Input state

        Returns:
            Action logits and state value
        """
        # Encode state classically
        encoded = self.encoder(state)

        # Split into amplitude and phase
        amplitude = encoded[:, : self.num_qubits]
        phase = encoded[:, self.num_qubits :]

        # Create quantum input
        quantum_input = torch.polar(amplitude, phase)

        # Quantum processing
        quantum_output = self.quantum_circuit(quantum_input)

        # Convert back to real values for classical processing
        classical_input = torch.cat([quantum_output.real, quantum_output.imag], dim=-1)

        # Decode to actions
        action_logits = self.decoder(classical_input)

        # Value estimation
        value = self.value_net(classical_input)

        return action_logits, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from policy.

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

    def get_quantum_parameters(self) -> List[torch.Tensor]:
        """Get quantum circuit parameters."""
        return list(self.quantum_circuit.parameters())

    def get_classical_parameters(self) -> List[torch.Tensor]:
        """Get classical network parameters."""
        classical_params = []
        classical_params.extend(self.encoder.parameters())
        classical_params.extend(self.decoder.parameters())
        classical_params.extend(self.value_net.parameters())
        return classical_params


class QuantumEnvironment:
    """
    Quantum-inspired environment for testing quantum RL algorithms.

    Simulates quantum mechanical systems as RL environments.
    """

    def __init__(self, num_qubits: int = 2, max_steps: int = 100):
        self.num_qubits = num_qubits
        self.max_steps = max_steps
        self.action_space = ["x", "y", "z", "h", "cnot", "measure"]
        self.observation_space_shape = (2**num_qubits * 2,)  # Real and imaginary parts

        # Quantum state
        self.state = torch.zeros(2**num_qubits, dtype=torch.complex64)
        self.state[0] = 1.0  # Start in |00...0⟩

        self.step_count = 0
        self.target_state = self._generate_target_state()

    def reset(self) -> torch.Tensor:
        """Reset environment."""
        self.state = torch.zeros(2**self.num_qubits, dtype=torch.complex64)
        self.state[0] = 1.0
        self.step_count = 0
        self.target_state = self._generate_target_state()

        return self._get_observation()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """Execute action in quantum environment."""
        self.step_count += 1

        # Apply quantum gate
        gate_name = self.action_space[action]
        self._apply_gate(gate_name)

        # Compute reward (fidelity with target state)
        fidelity = self._compute_fidelity(self.state, self.target_state)
        reward = fidelity.item()

        # Check termination
        terminated = fidelity > 0.95  # Close to target
        truncated = self.step_count >= self.max_steps

        observation = self._get_observation()
        info = {
            "fidelity": fidelity.item(),
            "gate_applied": gate_name,
            "target_state": self.target_state.detach().numpy(),
        }

        return observation, reward, terminated, truncated, info

    def _apply_gate(self, gate_name: str):
        """Apply quantum gate to current state."""
        if gate_name == "x":
            gate = PauliX()
            self.state = self._apply_single_qubit_gate(gate, 0)
        elif gate_name == "y":
            gate = PauliY()
            self.state = self._apply_single_qubit_gate(gate, 0)
        elif gate_name == "z":
            gate = PauliZ()
            self.state = self._apply_single_qubit_gate(gate, 0)
        elif gate_name == "h":
            gate = HadamardGate()
            self.state = self._apply_single_qubit_gate(gate, 0)
        elif gate_name == "cnot":
            self.state = self._apply_cnot(0, 1)
        elif gate_name == "measure":
            # Measurement collapses state
            probabilities = torch.abs(self.state) ** 2
            outcome = torch.multinomial(probabilities.real, 1).item()
            self.state = torch.zeros_like(self.state)
            self.state[outcome] = 1.0

    def _apply_single_qubit_gate(self, gate: QuantumGate, qubit: int) -> torch.Tensor:
        """Apply single-qubit gate."""
        num_states = 2**self.num_qubits
        op = torch.tensor([[1.0]], dtype=torch.complex64)

        for q in range(self.num_qubits):
            if q == qubit:
                op = torch.kron(op, gate.matrix)
            else:
                identity = torch.eye(2, dtype=torch.complex64)
                op = torch.kron(op, identity)

        return torch.matmul(op, self.state)

    def _apply_cnot(self, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate."""
        num_states = 2**self.num_qubits
        cnot_matrix = torch.zeros((num_states, num_states), dtype=torch.complex64)

        for i in range(num_states):
            bits = [(i >> q) & 1 for q in range(self.num_qubits)]
            if bits[control] == 1:
                bits[target] = 1 - bits[target]
            j = sum(bit << q for q, bit in enumerate(bits))
            cnot_matrix[j, i] = 1.0

        return torch.matmul(cnot_matrix, self.state)

    def _compute_fidelity(
        self, state1: torch.Tensor, state2: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantum state fidelity."""
        overlap = torch.abs(torch.conj(state1).T @ state2) ** 2
        return overlap

    def _generate_target_state(self) -> torch.Tensor:
        """Generate a random target quantum state."""
        # Create random state vector
        real_part = torch.randn(2**self.num_qubits)
        imag_part = torch.randn(2**self.num_qubits)
        state = torch.complex(real_part, imag_part)
        state = state / torch.norm(state)
        return state

    def _get_observation(self) -> torch.Tensor:
        """Get observation (quantum state representation)."""
        return torch.cat([self.state.real, self.state.imag])

    def render(self):
        """Render current quantum state."""
        print(f"Step: {self.step_count}")
        print(f"State: {self.state}")
        probabilities = torch.abs(self.state) ** 2
        print(f"Measurement probabilities: {probabilities}")

        # Show target fidelity
        fidelity = self._compute_fidelity(self.state, self.target_state)
        print(f"Target fidelity: {fidelity:.4f}")
