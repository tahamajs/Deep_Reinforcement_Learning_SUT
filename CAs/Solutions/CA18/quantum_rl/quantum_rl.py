"""
Quantum Reinforcement Learning Module

This module implements quantum-enhanced reinforcement learning algorithms,
including variational quantum circuits, quantum policy networks, and
quantum value function approximation.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any


class QuantumGate:
    """Quantum gate operations"""

    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli-X gate"""
        return np.array([[0, 1], [1, 0]], dtype=complex)

    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli-Y gate"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli-Z gate"""
        return np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def rotation_x(angle: float) -> np.ndarray:
        """Rotation around X-axis"""
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def rotation_y(angle: float) -> np.ndarray:
        """Rotation around Y-axis"""
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def rotation_z(angle: float) -> np.ndarray:
        """Rotation around Z-axis"""
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=complex)

    @staticmethod
    def cnot() -> Tuple[np.ndarray, List[int]]:
        """CNOT gate"""
        gate = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        return gate, [0, 1]  # control, target


class QuantumState:
    """Quantum state representation"""

    def __init__(self, amplitudes: np.ndarray):
        """Initialize quantum state with amplitudes"""
        self.amplitudes = np.array(amplitudes, dtype=complex)
        self.n_qubits = int(np.log2(len(amplitudes)))

        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm

    @classmethod
    def zero_state(cls, n_qubits: int) -> "QuantumState":
        """Create |0...0⟩ state"""
        amplitudes = np.zeros(2**n_qubits, dtype=complex)
        amplitudes[0] = 1.0
        return cls(amplitudes)

    @classmethod
    def uniform_superposition(cls, n_qubits: int) -> "QuantumState":
        """Create uniform superposition state"""
        n_states = 2**n_qubits
        amplitudes = np.ones(n_states, dtype=complex) / np.sqrt(n_states)
        return cls(amplitudes)

    def apply_gate(
        self, gate: np.ndarray, target_qubits: List[int] = None
    ) -> "QuantumState":
        """Apply quantum gate to state"""
        if target_qubits is None:
            target_qubits = list(range(self.n_qubits))

        if len(target_qubits) == 1:
            qubit = target_qubits[0]
            new_amplitudes = np.zeros_like(self.amplitudes, dtype=complex)

            for i in range(2**self.n_qubits):
                bit = (i >> qubit) & 1
                if bit == 0:
                    i1 = i | (1 << qubit)  # Set target bit to 1
                    new_amplitudes[i] = (
                        gate[0, 0] * self.amplitudes[i]
                        + gate[0, 1] * self.amplitudes[i1]
                    )
                else:
                    i0 = i & ~(1 << qubit)  # Set target bit to 0
                    new_amplitudes[i] = (
                        gate[1, 0] * self.amplitudes[i0]
                        + gate[1, 1] * self.amplitudes[i]
                    )
        else:
            new_amplitudes = gate @ self.amplitudes

        return QuantumState(new_amplitudes)

    def measure(self, qubit: int = 0) -> Tuple[int, "QuantumState"]:
        """Measure a qubit and collapse the state"""
        prob_0 = 0
        prob_1 = 0

        for i in range(len(self.amplitudes)):
            if (i >> qubit) & 1 == 0:
                prob_0 += abs(self.amplitudes[i]) ** 2
            else:
                prob_1 += abs(self.amplitudes[i]) ** 2

        if np.random.random() < prob_0:
            outcome = 0
            new_amplitudes = np.zeros_like(self.amplitudes)
            for i in range(len(self.amplitudes)):
                if (i >> qubit) & 1 == 0:
                    new_amplitudes[i] = self.amplitudes[i] / np.sqrt(prob_0)
        else:
            outcome = 1
            new_amplitudes = np.zeros_like(self.amplitudes)
            for i in range(len(self.amplitudes)):
                if (i >> qubit) & 1 == 1:
                    new_amplitudes[i] = self.amplitudes[i] / np.sqrt(prob_1)

        return outcome, QuantumState(new_amplitudes)

    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states"""
        return np.abs(self.amplitudes) ** 2

    def fidelity(self, other: "QuantumState") -> float:
        """Compute fidelity between two quantum states"""
        return abs(np.vdot(self.amplitudes, other.amplitudes)) ** 2


class QuantumCircuit:
    """Quantum circuit simulator"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = QuantumState.zero_state(n_qubits).amplitudes

    def reset(self):
        """Reset circuit to |0...0⟩ state"""
        self.state = QuantumState.zero_state(self.n_qubits).amplitudes

    def apply_single_gate(self, gate: np.ndarray, qubit: int):
        """Apply single-qubit gate"""
        quantum_state = QuantumState(self.state)
        new_state = quantum_state.apply_gate(gate, [qubit])
        self.state = new_state.amplitudes

    def apply_two_gate(self, gate: np.ndarray, control: int, target: int):
        """Apply two-qubit gate"""
        quantum_state = QuantumState(self.state)
        new_state = quantum_state.apply_gate(gate, [control, target])
        self.state = new_state.amplitudes

    def get_amplitudes(self) -> np.ndarray:
        """Get current state amplitudes"""
        return self.state.copy()

    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        return np.abs(self.state) ** 2


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

                    self.circuit.apply_single_gate(
                        QuantumGate.rotation_x(rx_angle), qubit
                    )
                    self.circuit.apply_single_gate(
                        QuantumGate.rotation_y(ry_angle), qubit
                    )
                    self.circuit.apply_single_gate(
                        QuantumGate.rotation_z(rz_angle), qubit
                    )

                    param_idx += 3

            elif self.gate_set == "ry":
                for qubit in range(self.n_qubits):
                    ry_angle = self.params[param_idx].item()
                    self.circuit.apply_single_gate(
                        QuantumGate.rotation_y(ry_angle), qubit
                    )
                    param_idx += 1

            if layer < self.n_layers - 1:  # No entanglement on last layer
                for qubit in range(self.n_qubits - 1):
                    cnot_gate, _ = QuantumGate.cnot()
                    self.circuit.apply_two_gate(cnot_gate, qubit, qubit + 1)

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
            circuit.apply_single_gate(QuantumGate.rotation_y(angle), i)

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

        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "quantum_gradients": [],
        }

    def get_action(self, state: torch.Tensor) -> np.ndarray:
        """Get action from quantum policy"""
        with torch.no_grad():
            action = self.policy(state)
            return action.squeeze().numpy()

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
    ) -> Dict[str, float]:
        """Single training step using quantum policy gradient"""

        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()

        targets = rewards + gamma * next_values * (1 - dones.float())
        advantages = targets - values

        value_loss = torch.nn.functional.mse_loss(values, targets.detach())

        policy_actions = self.policy(states)

        action_diff = torch.nn.functional.mse_loss(
            policy_actions, actions, reduction="none"
        )
        log_probs = -action_diff.sum(dim=-1)  # Simplified log-probability

        policy_loss = -(log_probs * advantages.detach()).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        quantum_grad_norm = 0.0
        for param in self.policy.vqc.parameters():
            if param.grad is not None:
                quantum_grad_norm += param.grad.norm().item() ** 2
        quantum_grad_norm = quantum_grad_norm**0.5

        self.policy_optimizer.step()

        self.training_stats["policy_loss"].append(policy_loss.item())
        self.training_stats["value_loss"].append(value_loss.item())
        self.training_stats["quantum_gradients"].append(quantum_grad_norm)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "quantum_grad_norm": quantum_grad_norm,
        }


class QuantumQLearning:
    """Quantum Q-Learning implementation"""

    def __init__(
        self,
        n_qubits: int,
        n_actions: int,
        n_layers: int = 3,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
    ):

        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.q_circuits = {}
        for action in range(n_actions):
            self.q_circuits[action] = VariationalQuantumCircuit(n_qubits, n_layers)

        self.q_observable = QuantumGate.pauli_z()

    def state_to_quantum(self, state: np.ndarray) -> QuantumState:
        """Encode classical state to quantum state"""
        if len(state) <= 2**self.n_qubits:
            amplitudes = np.zeros(2**self.n_qubits)
            amplitudes[: len(state)] = state
            amplitudes = amplitudes / np.linalg.norm(amplitudes)
            return QuantumState(amplitudes)
        else:
            state_index = int(np.sum(state * [2**i for i in range(len(state))]))
            state_index = state_index % (2**self.n_qubits)
            amplitudes = np.zeros(2**self.n_qubits)
            amplitudes[state_index] = 1.0
            return QuantumState(amplitudes)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions"""
        quantum_state = self.state_to_quantum(state)
        q_values = np.zeros(self.n_actions)

        for action in range(self.n_actions):
            q_values[action] = self.q_circuits[action].measure_expectation(
                self.q_observable, quantum_state
            )

        return q_values

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update quantum Q-function"""
        quantum_state = self.state_to_quantum(state)

        current_q = self.q_circuits[action].measure_expectation(
            self.q_observable, quantum_state
        )

        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target_q = reward + self.gamma * np.max(next_q_values)

        td_error = target_q - current_q

        for param_idx in range(self.q_circuits[action].n_parameters):
            gradient = self.q_circuits[action].gradient(
                self.q_observable, quantum_state, param_idx
            )

            self.q_circuits[action].parameters[param_idx] += (
                self.learning_rate * td_error * gradient
            )


class QuantumActorCritic:
    """Quantum Actor-Critic implementation"""

    def __init__(self, n_qubits: int, n_actions: int, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_actions = n_actions

        self.actor_circuit = VariationalQuantumCircuit(n_qubits, n_layers)
        self.critic_circuit = VariationalQuantumCircuit(n_qubits, n_layers)

        self.policy_observables = [QuantumGate.pauli_z() for _ in range(n_actions)]
        self.value_observable = QuantumGate.pauli_z()

        self.learning_rate = 0.01
        self.gamma = 0.95

    def state_to_quantum(self, state: np.ndarray) -> QuantumState:
        """Convert classical state to quantum state"""
        amplitudes = np.zeros(2**self.n_qubits)
        state_norm = np.linalg.norm(state)
        if state_norm > 0:
            state = state / state_norm

        for i, val in enumerate(state[: 2**self.n_qubits]):
            amplitudes[i] = val

        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        return QuantumState(amplitudes)

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities from quantum actor"""
        quantum_state = self.state_to_quantum(state)

        expectations = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            expectations[action] = self.actor_circuit.measure_expectation(
                self.policy_observables[action], quantum_state
            )

        exp_vals = np.exp(expectations)
        probabilities = exp_vals / np.sum(exp_vals)

        return probabilities

    def get_value(self, state: np.ndarray) -> float:
        """Get state value from quantum critic"""
        quantum_state = self.state_to_quantum(state)
        return self.critic_circuit.measure_expectation(
            self.value_observable, quantum_state
        )

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from quantum policy"""
        probabilities = self.get_action_probabilities(state)
        return np.random.choice(self.n_actions, p=probabilities)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update actor and critic"""
        quantum_state = self.state_to_quantum(state)

        current_value = self.get_value(state)
        if done:
            target_value = reward
        else:
            next_value = self.get_value(next_state)
            target_value = reward + self.gamma * next_value

        td_error = target_value - current_value

        for param_idx in range(self.critic_circuit.n_parameters):
            gradient = self.critic_circuit.gradient(
                self.value_observable, quantum_state, param_idx
            )
            self.critic_circuit.parameters[param_idx] += (
                self.learning_rate * td_error * gradient
            )

        for param_idx in range(self.actor_circuit.n_parameters):
            gradient = self.actor_circuit.gradient(
                self.policy_observables[action], quantum_state, param_idx
            )
            self.actor_circuit.parameters[param_idx] += (
                self.learning_rate * td_error * gradient
            )


class QuantumEnvironment:
    """Simple quantum-inspired environment"""

    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.state_dim = 2**n_qubits
        self.n_actions = 4  # Four possible quantum gates

        self.target_state = QuantumState.uniform_superposition(n_qubits)
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_state = QuantumState.zero_state(self.n_qubits)
        self.steps = 0
        return self.current_state.amplitudes.real

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Environment step"""
        if action == 0:  # Hadamard on first qubit
            gate = QuantumGate.hadamard()
        elif action == 1:  # Pauli-X on first qubit
            gate = QuantumGate.pauli_x()
        elif action == 2:  # Rotation-Y
            gate = QuantumGate.rotation_y(np.pi / 4)
        else:  # Rotation-Z
            gate = QuantumGate.rotation_z(np.pi / 4)

        self.current_state = self.current_state.apply_gate(gate, [0])

        fidelity = self.current_state.fidelity(self.target_state)

        reward = fidelity
        self.steps += 1
        done = self.steps >= 10 or fidelity > 0.95

        return self.current_state.amplitudes.real, reward, done, {}
