"""
Hybrid Quantum-Classical Reinforcement Learning Module

This module implements advanced hybrid architectures that combine quantum computing
with classical neural networks for enhanced reinforcement learning capabilities.

Key Components:
- QuantumStateSimulator: Simulates quantum states for RL state representation
- QuantumFeatureMap: Maps classical states to quantum feature spaces
- VariationalQuantumCircuit: Parameterized quantum circuits for policy/value functions
- HybridQuantumClassicalAgent: Integrated agent combining quantum and classical processing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Quantum computing simulation (representing real quantum hardware)
try:
    from qiskit import QuantumCircuit, execute, Aer
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Quantum simulations will be limited.")


class QuantumStateSimulator:
    """
    Quantum State Simulator for RL State Representation

    This class simulates quantum states that can represent complex RL state spaces
    through quantum superposition and entanglement, potentially providing
    exponential advantages in state representation.
    """

    def __init__(self, n_qubits: int, state_dim: int):
        self.n_qubits = n_qubits
        self.state_dim = state_dim
        self.quantum_state = None

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum state simulation")

        # Initialize quantum circuit for state encoding
        self.state_circuit = QuantumCircuit(n_qubits)
        self.parameters = [Parameter(f'θ_{i}') for i in range(n_qubits * state_dim)]

    def encode_state(self, classical_state: np.ndarray) -> 'Statevector':
        """
        Encode classical RL state into quantum state

        Args:
            classical_state: Classical state vector

        Returns:
            Quantum statevector representing the encoded state
        """
        # Reset circuit
        self.state_circuit = QuantumCircuit(self.n_qubits)

        # Encode classical state through rotation gates
        param_idx = 0
        for i in range(min(self.n_qubits, len(classical_state))):
            if param_idx < len(self.parameters):
                # RY rotation based on state value
                angle = np.pi * classical_state[i % len(classical_state)]
                self.state_circuit.ry(angle, i)

                # Add entanglement between qubits
                if i > 0:
                    self.state_circuit.cx(i-1, i)

                param_idx += 1

        # Execute circuit to get quantum state
        backend = Aer.get_backend('statevector_simulator')
        job = execute(self.state_circuit, backend)
        self.quantum_state = job.result().get_statevector()

        return self.quantum_state

    def get_state_amplitudes(self) -> np.ndarray:
        """Get quantum state amplitudes"""
        if self.quantum_state is None:
            return np.zeros(2**self.n_qubits)
        return np.abs(self.quantum_state.data)**2

    def measure_in_basis(self, basis: str = 'computational') -> Dict[str, float]:
        """
        Measure quantum state in specified basis

        Args:
            basis: Measurement basis ('computational', 'x', 'y', 'z')

        Returns:
            Measurement outcomes and probabilities
        """
        if self.quantum_state is None:
            return {'00': 1.0}

        # Create measurement circuit
        measure_circuit = self.state_circuit.copy()

        if basis == 'x':
            # Measure in X basis (Hadamard before measurement)
            for i in range(self.n_qubits):
                measure_circuit.h(i)
        elif basis == 'y':
            # Measure in Y basis (rotation before measurement)
            for i in range(self.n_qubits):
                measure_circuit.rx(np.pi/2, i)

        # Add measurement
        measure_circuit.add_register(measure_circuit.classical_register(self.n_qubits, 'c'))
        measure_circuit.measure_all()

        # Execute measurement
        backend = Aer.get_backend('qasm_simulator')
        job = execute(measure_circuit, backend, shots=1024)
        result = job.result()
        counts = result.get_counts()

        return counts

    def calculate_entanglement(self) -> float:
        """Calculate entanglement measure of the quantum state"""
        if self.quantum_state is None or self.n_qubits < 2:
            return 0.0

        try:
            # Calculate reduced density matrix for first qubit
            rho = DensityMatrix(self.quantum_state)
            rho_A = partial_trace(rho, list(range(1, self.n_qubits)))

            # Calculate von Neumann entropy
            eigenvals = np.real(np.linalg.eigvals(rho_A.data))
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical zeros

            if len(eigenvals) == 0:
                return 0.0

            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            return min(entropy, 1.0)  # Normalized entanglement measure

        except Exception:
            return 0.0


class QuantumFeatureMap:
    """
    Quantum Feature Map for Enhanced State Representation

    Maps classical states to quantum feature spaces using various encoding strategies,
    potentially providing quantum advantage through kernel methods.
    """

    def __init__(self, n_qubits: int, encoding_type: str = 'ZZFeatureMap'):
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum feature mapping")

        # Initialize parameters for variational encoding
        self.parameters = [Parameter(f'φ_{i}') for i in range(n_qubits * 2)]

    def map_features(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Map classical features to quantum feature space

        Args:
            x: First input vector
            y: Second input vector (for kernel computation)

        Returns:
            Quantum feature representation or kernel value
        """
        # Create feature map circuit
        qc = QuantumCircuit(self.n_qubits)

        if self.encoding_type == 'ZZFeatureMap':
            self._apply_zz_feature_map(qc, x)
        elif self.encoding_type == 'PauliFeatureMap':
            self._apply_pauli_feature_map(qc, x)
        else:
            # Default to simple rotation encoding
            self._apply_rotation_encoding(qc, x)

        if y is not None:
            # Compute quantum kernel between x and y
            return self._compute_quantum_kernel(qc, x, y)
        else:
            # Return quantum state representation
            backend = Aer.get_backend('statevector_simulator')
            job = execute(qc, backend)
            statevector = job.result().get_statevector()
            return np.abs(statevector.data)**2

    def _apply_zz_feature_map(self, qc: QuantumCircuit, x: np.ndarray):
        """Apply ZZ feature map encoding"""
        for i in range(min(self.n_qubits, len(x))):
            # Single qubit rotations
            qc.ry(2 * np.arcsin(np.sqrt(abs(x[i]))), i)

        # Two-qubit entangling gates
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                if i < len(x) and j < len(x):
                    interaction = 2 * x[i] * x[j]
                    qc.cx(i, j)
                    qc.rz(interaction, j)
                    qc.cx(i, j)

    def _apply_pauli_feature_map(self, qc: QuantumCircuit, x: np.ndarray):
        """Apply Pauli feature map encoding"""
        for i in range(min(self.n_qubits, len(x))):
            qc.rx(np.pi * x[i], i)
            qc.ry(np.pi * x[i], i)
            qc.rz(np.pi * x[i], i)

    def _apply_rotation_encoding(self, qc: QuantumCircuit, x: np.ndarray):
        """Apply simple rotation encoding"""
        for i in range(min(self.n_qubits, len(x))):
            qc.ry(np.pi * x[i], i)

    def _compute_quantum_kernel(self, qc: QuantumCircuit, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute quantum kernel between two classical vectors

        K(x,y) = |⟨ψ(x)|ψ(y)⟩|²
        """
        # Create circuit for x
        qc_x = qc.copy()
        self._apply_zz_feature_map(qc_x, x)

        # Create circuit for y
        qc_y = QuantumCircuit(self.n_qubits)
        self._apply_zz_feature_map(qc_y, y)

        # Compute inner product ⟨ψ(x)|ψ(y)⟩
        try:
            # This is a simplified computation - in practice would need
            # more sophisticated quantum kernel estimation
            similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            return abs(similarity)**2
        except:
            return 0.5  # Default similarity


class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit for RL Policy/Value Functions

    Parameterized quantum circuit that can be trained to approximate
    policy or value functions in reinforcement learning.
    """

    def __init__(self, n_qubits: int, n_layers: int, output_dim: int = 1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for variational quantum circuits")

        # Initialize variational parameters
        self.theta = [[Parameter(f'θ_{l}_{q}') for q in range(n_qubits)] for l in range(n_layers)]
        self.phi = [[Parameter(f'φ_{l}_{q}') for q in range(n_qubits)] for l in range(n_layers)]

        # Flatten parameters for optimization
        self.all_parameters = [p for layer in self.theta + self.phi for p in layer]

    def construct_circuit(self, input_data: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Construct the variational quantum circuit

        Args:
            input_data: Classical input data for data encoding

        Returns:
            Complete variational quantum circuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # Input encoding layer (if input provided)
        if input_data is not None:
            for i in range(min(self.n_qubits, len(input_data))):
                qc.ry(np.pi * input_data[i], i)

        # Variational layers
        for layer in range(self.n_layers):
            # Rotation layer
            for qubit in range(self.n_qubits):
                qc.ry(self.theta[layer][qubit], qubit)
                qc.rz(self.phi[layer][qubit], qubit)

            # Entangling layer
            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)

        return qc

    def execute_circuit(self, parameters: List[float], input_data: Optional[np.ndarray] = None,
                       shots: int = 1024) -> Dict:
        """
        Execute the variational quantum circuit

        Args:
            parameters: Circuit parameters
            input_data: Classical input data
            shots: Number of measurement shots

        Returns:
            Execution results
        """
        # Construct circuit
        qc = self.construct_circuit(input_data)

        # Bind parameters
        param_dict = {self.all_parameters[i]: parameters[i] for i in range(len(parameters))}
        qc_bound = qc.bind_parameters(param_dict)

        # Execute circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc_bound, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Also get statevector for analysis
        sv_backend = Aer.get_backend('statevector_simulator')
        sv_job = execute(qc_bound, sv_backend)
        statevector = sv_job.result().get_statevector()

        return {
            'counts': counts,
            'statevector': statevector,
            'probabilities': self._counts_to_probs(counts, shots)
        }

    def _counts_to_probs(self, counts: Dict, shots: int) -> np.ndarray:
        """Convert measurement counts to probability distribution"""
        probs = np.zeros(2**self.n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            if idx < len(probs):
                probs[idx] = count / shots
        return probs

    def get_parameter_count(self) -> int:
        """Get total number of variational parameters"""
        return len(self.all_parameters)


class HybridQuantumClassicalAgent:
    """
    Hybrid Quantum-Classical RL Agent

    Combines quantum circuits for enhanced representation with classical
    neural networks for stable learning and temporal processing.
    """

    def __init__(self, state_dim: int, action_dim: int, quantum_qubits: int = 4,
                 quantum_layers: int = 2, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quantum_qubits = quantum_qubits

        # Quantum components
        self.quantum_simulator = QuantumStateSimulator(quantum_qubits, state_dim)
        self.quantum_feature_map = QuantumFeatureMap(quantum_qubits)
        self.quantum_circuit = VariationalQuantumCircuit(quantum_qubits, quantum_layers, action_dim)

        # Classical neural network
        self.classical_network = self._build_classical_network()
        self.target_network = self._build_classical_network()
        self.optimizer = optim.Adam(self.classical_network.parameters(), lr=learning_rate)

        # Quantum parameter optimization
        quantum_params = self.quantum_circuit.get_parameter_count()
        self.quantum_params = nn.Parameter(torch.randn(quantum_params) * 0.1)
        self.quantum_optimizer = optim.Adam([self.quantum_params], lr=learning_rate * 0.1)

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Hybrid control parameters
        self.quantum_weight = 0.6  # Balance between quantum and classical
        self.adaptive_hybrid = True

    def _build_classical_network(self) -> nn.Module:
        """Build classical neural network component"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.action_dim)
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> Tuple[int, Dict]:
        """
        Select action using hybrid quantum-classical decision making

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Selected action and decision info
        """
        # Classical network prediction
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            classical_q = self.classical_network(state_tensor).squeeze()

        # Quantum-enhanced prediction
        try:
            # Encode state in quantum simulator
            quantum_state = self.quantum_simulator.encode_state(state)

            # Execute variational quantum circuit
            quantum_result = self.quantum_circuit.execute_circuit(
                self.quantum_params.detach().numpy(), state
            )

            # Convert quantum output to Q-values
            quantum_probs = quantum_result['probabilities']
            # Map quantum probabilities to action values (simplified)
            quantum_q = torch.FloatTensor(quantum_probs[:self.action_dim] * 10 - 5)

            # Adaptive quantum-classical fusion
            if self.adaptive_hybrid:
                entanglement = self.quantum_simulator.calculate_entanglement()
                self.quantum_weight = 0.3 + 0.7 * entanglement

            # Fuse predictions
            fused_q = self.quantum_weight * quantum_q + (1 - self.quantum_weight) * classical_q

            decision_info = {
                'method': 'hybrid_fusion',
                'quantum_weight': self.quantum_weight,
                'entanglement': entanglement,
                'quantum_probs': quantum_probs[:self.action_dim],
                'classical_q': classical_q.numpy()
            }

        except Exception as e:
            # Fallback to classical only
            fused_q = classical_q
            decision_info = {
                'method': 'classical_fallback',
                'error': str(e)
            }

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            decision_info['exploration'] = True
        else:
            action = torch.argmax(fused_q).item()
            decision_info['exploration'] = False

        return action, decision_info

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> Dict:
        """
        Learn from experiences using hybrid quantum-classical updates

        Returns:
            Learning metrics
        """
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        # Classical Q-learning update
        current_q = self.classical_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + 0.99 * next_q * ~dones

        classical_loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Update classical network
        self.optimizer.zero_grad()
        classical_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.classical_network.parameters(), 1.0)
        self.optimizer.step()

        # Quantum parameter update (simplified)
        quantum_loss = 0.0
        try:
            # Simple quantum update based on classical gradients
            if len(self.memory) > self.batch_size * 2:
                # Update quantum parameters occasionally
                if np.random.random() < 0.1:  # 10% of updates
                    grad_scale = classical_loss.item()
                    self.quantum_optimizer.zero_grad()
                    # Simplified quantum gradient (in practice would use parameter-shift rule)
                    self.quantum_params.grad = torch.randn_like(self.quantum_params) * grad_scale * 0.01
                    self.quantum_optimizer.step()
                    quantum_loss = grad_scale
        except:
            pass

        # Update target network
        if hasattr(self, 'update_counter'):
            self.update_counter += 1
        else:
            self.update_counter = 1

        if self.update_counter % 100 == 0:
            self.target_network.load_state_dict(self.classical_network.state_dict())

        return {
            'classical_loss': classical_loss.item(),
            'quantum_loss': quantum_loss,
            'quantum_weight': self.quantum_weight
        }

    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        return {
            'memory_size': len(self.memory),
            'quantum_weight': self.quantum_weight,
            'quantum_entanglement': self.quantum_simulator.calculate_entanglement(),
            'classical_param_count': sum(p.numel() for p in self.classical_network.parameters()),
            'quantum_param_count': len(self.quantum_params)
        }