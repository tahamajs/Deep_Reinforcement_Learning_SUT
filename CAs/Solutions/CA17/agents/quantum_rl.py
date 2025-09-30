import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import cmath
import random


class QuantumGate(ABC):
    """Base class for quantum gates"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the unitary matrix representation of the gate"""
        pass

    def __str__(self):
        return f"{self.name} gate"


class PauliX(QuantumGate):
    """Pauli-X (NOT) gate"""

    def __init__(self):
        super().__init__("Pauli-X")

    def matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)


class PauliY(QuantumGate):
    """Pauli-Y gate"""

    def __init__(self):
        super().__init__("Pauli-Y")

    def matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)


class PauliZ(QuantumGate):
    """Pauli-Z gate"""

    def __init__(self):
        super().__init__("Pauli-Z")

    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=complex)


class Hadamard(QuantumGate):
    """Hadamard gate"""

    def __init__(self):
        super().__init__("Hadamard")

    def matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


class RotationX(QuantumGate):
    """Rotation around X-axis"""

    def __init__(self, angle: float):
        super().__init__(f"RotationX({angle:.3f})")
        self.angle = angle

    def matrix(self) -> np.ndarray:
        cos = np.cos(self.angle / 2)
        sin = np.sin(self.angle / 2)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=complex)


class RotationY(QuantumGate):
    """Rotation around Y-axis"""

    def __init__(self, angle: float):
        super().__init__(f"RotationY({angle:.3f})")
        self.angle = angle

    def matrix(self) -> np.ndarray:
        cos = np.cos(self.angle / 2)
        sin = np.sin(self.angle / 2)
        return np.array([[cos, -sin], [sin, cos]], dtype=complex)


class RotationZ(QuantumGate):
    """Rotation around Z-axis"""

    def __init__(self, angle: float):
        super().__init__(f"RotationZ({angle:.3f})")
        self.angle = angle

    def matrix(self) -> np.ndarray:
        cos = np.cos(self.angle / 2)
        sin = np.sin(self.angle / 2)
        return np.array([[cos - 1j * sin, 0], [0, cos + 1j * sin]], dtype=complex)


class CNOT(QuantumGate):
    """Controlled-NOT gate"""

    def __init__(self):
        super().__init__("CNOT")

    def matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )


class QuantumCircuit:
    """Basic quantum circuit simulator"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits

        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0

        self.gate_history = []

    def reset(self):
        """Reset circuit to |0...0⟩ state"""
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0
        self.gate_history = []

    def apply_single_gate(self, gate: QuantumGate, qubit: int):
        """Apply single-qubit gate to specified qubit"""
        if qubit >= self.n_qubits:
            raise ValueError(
                f"Qubit {qubit} out of range for {self.n_qubits}-qubit circuit"
            )

        full_matrix = self._expand_gate_matrix(gate.matrix(), qubit)

        self.state = full_matrix @ self.state

        self.gate_history.append((gate.name, qubit))

    def apply_two_gate(self, gate: QuantumGate, control_qubit: int, target_qubit: int):
        """Apply two-qubit gate"""
        if max(control_qubit, target_qubit) >= self.n_qubits:
            raise ValueError("Qubit index out of range")

        full_matrix = self._expand_two_gate_matrix(
            gate.matrix(), control_qubit, target_qubit
        )

        self.state = full_matrix @ self.state

        self.gate_history.append((gate.name, control_qubit, target_qubit))

    def _expand_gate_matrix(
        self, gate_matrix: np.ndarray, target_qubit: int
    ) -> np.ndarray:
        """Expand single-qubit gate to full circuit matrix"""
        n = self.n_qubits
        full_matrix = np.eye(self.n_states, dtype=complex)

        for i in range(self.n_states):
            for j in range(self.n_states):
                if self._differ_only_at_qubit(i, j, target_qubit):
                    bit_i = (i >> target_qubit) & 1
                    bit_j = (j >> target_qubit) & 1
                    full_matrix[i, j] = gate_matrix[bit_i, bit_j]
                elif i != j:
                    full_matrix[i, j] = 0

        return full_matrix

    def _expand_two_gate_matrix(
        self, gate_matrix: np.ndarray, control_qubit: int, target_qubit: int
    ) -> np.ndarray:
        """Expand two-qubit gate to full circuit matrix"""
        n = self.n_qubits
        full_matrix = np.eye(self.n_states, dtype=complex)

        for i in range(self.n_states):
            for j in range(self.n_states):
                control_bit_i = (i >> control_qubit) & 1
                target_bit_i = (i >> target_qubit) & 1
                control_bit_j = (j >> control_qubit) & 1
                target_bit_j = (j >> target_qubit) & 1

                if control_bit_i == control_bit_j:
                    if control_bit_i == 1:  # Control is |1⟩
                        if target_bit_i == 0 and target_bit_j == 1:
                            full_matrix[i, j] = 1
                        elif target_bit_i == 1 and target_bit_j == 0:
                            full_matrix[i, j] = 1
                        else:
                            full_matrix[i, j] = 0
                    else:  # Control is |0⟩, identity on target
                        if i == j:
                            full_matrix[i, j] = 1
                        else:
                            full_matrix[i, j] = 0
                else:
                    full_matrix[i, j] = 0

        return full_matrix

    def _differ_only_at_qubit(self, i: int, j: int, qubit: int) -> bool:
        """Check if states i and j differ only at the specified qubit"""
        mask = 1 << qubit
        return (i & ~mask) == (j & ~mask)

    def measure(self, qubit: int) -> int:
        """Measure specified qubit"""
        prob_0 = 0
        for i in range(self.n_states):
            if ((i >> qubit) & 1) == 0:
                prob_0 += abs(self.state[i]) ** 2

        if np.random.random() < prob_0:
            return 0
        else:
            return 1

    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        return np.abs(self.state) ** 2

    def get_amplitudes(self) -> np.ndarray:
        """Get state amplitudes"""
        return self.state.copy()


class VariationalQuantumCircuit(nn.Module):
    """Parameterized quantum circuit for quantum machine learning"""

    def __init__(self, n_qubits: int, n_layers: int, gate_set: str = "full"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.gate_set = gate_set

        if gate_set == "full":
            n_params_per_layer = 3 * n_qubits
        elif gate_set == "ry":
            n_params_per_layer = n_qubits
        else:
            n_params_per_layer = n_qubits

        self.n_params = n_params_per_layer * n_layers

        self.params = nn.Parameter(torch.randn(self.n_params) * 0.1)

        self.circuit = QuantumCircuit(n_qubits)

    def forward(self, input_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute variational quantum circuit"""
        self.circuit.reset()

        if input_state is not None:
            self.circuit.state = input_state.astype(complex)

        param_idx = 0

        for layer in range(self.n_layers):
            if self.gate_set == "full":
                for qubit in range(self.n_qubits):
                    rx_angle = self.params[param_idx].item()
                    ry_angle = self.params[param_idx + 1].item()
                    rz_angle = self.params[param_idx + 2].item()

                    self.circuit.apply_single_gate(RotationX(rx_angle), qubit)
                    self.circuit.apply_single_gate(RotationY(ry_angle), qubit)
                    self.circuit.apply_single_gate(RotationZ(rz_angle), qubit)

                    param_idx += 3

            elif self.gate_set == "ry":
                for qubit in range(self.n_qubits):
                    ry_angle = self.params[param_idx].item()
                    self.circuit.apply_single_gate(RotationY(ry_angle), qubit)
                    param_idx += 1

            if layer < self.n_layers - 1:  # No entanglement on last layer
                for qubit in range(self.n_qubits - 1):
                    self.circuit.apply_two_gate(CNOT(), qubit, qubit + 1)

        return self.circuit.get_amplitudes()

    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        amplitudes = self.forward()
        return np.abs(amplitudes) ** 2

    def measure_expectation(self, observable: np.ndarray) -> float:
        """Measure expectation value of observable"""
        state = self.forward()
        return np.real(np.conj(state) @ observable @ state)


class QuantumStateEncoder:
    """Encode classical data into quantum states"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits

    def amplitude_encoding(self, data: np.ndarray) -> np.ndarray:
        """Encode data as quantum amplitudes"""
        data = data.real.astype(float)  # Ensure real

        if len(data) > self.n_states:
            data = data[: self.n_states]
        elif len(data) < self.n_states:
            padded_data = np.zeros(self.n_states)
            padded_data[: len(data)] = data
            data = padded_data

        norm = np.linalg.norm(data)
        if norm > 0:
            data = data / norm
        else:
            data = np.zeros_like(data)
            data[0] = 1.0  # Default to |0...0⟩

        return data.astype(complex)

    def angle_encoding(self, data: np.ndarray) -> np.ndarray:
        """Encode data using rotation angles"""
        circuit = QuantumCircuit(self.n_qubits)

        for i, angle in enumerate(data[: self.n_qubits]):
            circuit.apply_single_gate(RotationY(angle), i)

        return circuit.get_amplitudes()


class QuantumPolicy(nn.Module):
    """Quantum policy using variational quantum circuit"""

    def __init__(
        self, state_dim: int, action_dim: int, n_qubits: int = 4, n_layers: int = 3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.state_encoder = nn.Linear(state_dim, min(2**n_qubits, 16))

        self.vqc = VariationalQuantumCircuit(n_qubits, n_layers, "ry")

        self.quantum_encoder = QuantumStateEncoder(n_qubits)

        self.action_decoder = nn.Sequential(
            nn.Linear(2**n_qubits, 32), nn.ReLU(), nn.Linear(32, action_dim), nn.Tanh()
        )

        self.observables = []
        for i in range(action_dim):
            obs = np.eye(2**n_qubits, dtype=complex)
            qubit_idx = i % n_qubits
            for j in range(2**n_qubits):
                if (j >> qubit_idx) & 1:  # If qubit is |1⟩
                    obs[j, j] = -1.0
            self.observables.append(obs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        actions = []

        for b in range(batch_size):
            encoded_state = self.state_encoder(state[b : b + 1])
            encoded_state = torch.tanh(encoded_state).squeeze().detach().numpy()

            quantum_state = self.quantum_encoder.amplitude_encoding(encoded_state)

            output_state = self.vqc(quantum_state)

            action_values = []
            for obs in self.observables:
                expectation = np.real(np.conj(output_state) @ obs @ output_state)
                action_values.append(expectation)

            actions.append(action_values)

        return torch.FloatTensor(actions)


class QuantumValueNetwork(nn.Module):
    """Quantum value function approximator"""

    def __init__(self, state_dim: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.n_qubits = n_qubits

        self.state_encoder = nn.Linear(state_dim, min(2**n_qubits, 8))

        self.vqc = VariationalQuantumCircuit(n_qubits, n_layers, "ry")

        self.quantum_encoder = QuantumStateEncoder(n_qubits)

        self.value_observable = np.eye(2**n_qubits, dtype=complex)
        for i in range(2**n_qubits):
            if i & 1:  # If first qubit is |1⟩
                self.value_observable[i, i] = -1.0

        self.value_scale = nn.Parameter(torch.tensor(1.0))
        self.value_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        values = []

        for b in range(batch_size):
            encoded_state = self.state_encoder(state[b : b + 1])
            encoded_state = torch.tanh(encoded_state).squeeze().detach().numpy()

            quantum_state = self.quantum_encoder.amplitude_encoding(encoded_state)

            output_state = self.vqc(quantum_state)

            value_expectation = np.real(
                np.conj(output_state) @ self.value_observable @ output_state
            )

            scaled_value = self.value_scale * value_expectation + self.value_bias
            values.append(scaled_value.item())

        return torch.FloatTensor(values).unsqueeze(-1)


class QuantumRLAgent:
    """Quantum-enhanced reinforcement learning agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_qubits: int = 4,
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits

        self.policy = QuantumPolicy(state_dim, action_dim, n_qubits)
        self.value_net = QuantumValueNetwork(state_dim, n_qubits)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )

        self.replay_buffer = []

        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "quantum_gradients": [],
        }

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using quantum policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(state_tensor)
            return action.squeeze().numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

    def train_step(self) -> Dict[str, float]:
        """Train the agent for one step"""
        if len(self.replay_buffer) < 32:
            return {}

        # Sample batch
        batch = random.sample(self.replay_buffer, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Train step
        metrics = self.train_step_batch(states, actions, rewards, next_states, dones)

        return {"quantum_entropy": 0.5}  # Placeholder
