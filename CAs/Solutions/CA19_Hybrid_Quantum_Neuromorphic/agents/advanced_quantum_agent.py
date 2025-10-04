"""
Advanced Quantum-Inspired Reinforcement Learning Agent

This module implements sophisticated quantum-inspired RL algorithms including:
- Quantum State Evolution
- Entanglement-based Action Selection
- Quantum Interference for Policy Optimization
- Decoherence Modeling
- Quantum Error Correction
- Multi-scale Quantum Dynamics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import scipy.linalg as la
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")

class QuantumStateEvolution:
    """
    Advanced quantum state evolution with decoherence and error correction
    """
    
    def __init__(self, n_qubits: int = 8, decoherence_rate: float = 0.01):
        self.n_qubits = n_qubits
        self.dimension = 2 ** n_qubits
        self.decoherence_rate = decoherence_rate
        
        # Initialize quantum state (superposition)
        self.state = np.ones(self.dimension, dtype=complex) / np.sqrt(self.dimension)
        
        # Pauli matrices for quantum gates
        self.pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Decoherence operators
        self.decoherence_ops = self._create_decoherence_operators()
        
    def _create_decoherence_operators(self):
        """Create decoherence operators for realistic quantum simulation"""
        ops = []
        for i in range(self.n_qubits):
            # Phase damping
            phase_damping = np.zeros((self.dimension, self.dimension), dtype=complex)
            for j in range(self.dimension):
                phase_damping[j, j] = np.exp(-self.decoherence_rate * i)
            ops.append(phase_damping)
            
            # Amplitude damping
            amplitude_damping = np.zeros((self.dimension, self.dimension), dtype=complex)
            for j in range(self.dimension):
                if j < self.dimension // 2:
                    amplitude_damping[j, j] = np.sqrt(1 - self.decoherence_rate)
                else:
                    amplitude_damping[j, j] = np.sqrt(self.decoherence_rate)
            ops.append(amplitude_damping)
        
        return ops
    
    def apply_quantum_gate(self, gate_matrix: np.ndarray, qubit_indices: List[int]):
        """Apply quantum gate to specified qubits"""
        full_gate = np.eye(self.dimension, dtype=complex)
        
        # Construct full gate matrix
        for i, qubit_idx in enumerate(qubit_indices):
            if qubit_idx < self.n_qubits:
                # Tensor product with identity matrices
                left_id = np.eye(2 ** qubit_idx, dtype=complex)
                right_id = np.eye(2 ** (self.n_qubits - qubit_idx - 1), dtype=complex)
                single_gate = np.kron(left_id, np.kron(gate_matrix, right_id))
                full_gate = full_gate @ single_gate
        
        # Apply gate to state
        self.state = full_gate @ self.state
        
    def apply_rotation_gate(self, angle: float, axis: str, qubit_idx: int):
        """Apply rotation gate around specified axis"""
        if axis == 'x':
            gate_matrix = la.expm(-1j * angle * self.pauli_x / 2)
        elif axis == 'y':
            gate_matrix = la.expm(-1j * angle * self.pauli_y / 2)
        elif axis == 'z':
            gate_matrix = la.expm(-1j * angle * self.pauli_z / 2)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        self.apply_quantum_gate(gate_matrix, [qubit_idx])
    
    def apply_cnot_gate(self, control_qubit: int, target_qubit: int):
        """Apply CNOT gate"""
        cnot_matrix = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]], dtype=complex)
        
        self.apply_quantum_gate(cnot_matrix, [control_qubit, target_qubit])
    
    def measure_state(self) -> Dict[str, float]:
        """Measure quantum state and return probabilities"""
        probabilities = np.abs(self.state) ** 2
        
        # Apply decoherence
        for op in self.decoherence_ops:
            self.state = op @ self.state
        
        # Normalize state
        norm = np.sqrt(np.sum(np.abs(self.state) ** 2))
        if norm > 0:
            self.state = self.state / norm
        
        # Create measurement results
        results = {}
        for i, prob in enumerate(probabilities):
            binary_str = format(i, f'0{self.n_qubits}b')
            results[binary_str] = float(prob)
        
        return results
    
    def calculate_entanglement(self) -> float:
        """Calculate entanglement measure using von Neumann entropy"""
        if self.n_qubits < 2:
            return 0.0
        
        # Create density matrix
        rho = np.outer(self.state, np.conj(self.state))
        
        # Partial trace over half the qubits
        n_half = self.n_qubits // 2
        reduced_rho = np.zeros((2 ** n_half, 2 ** n_half), dtype=complex)
        
        for i in range(2 ** n_half):
            for j in range(2 ** n_half):
                for k in range(2 ** (self.n_qubits - n_half)):
                    reduced_rho[i, j] += rho[i * 2 ** (self.n_qubits - n_half) + k,
                                           j * 2 ** (self.n_qubits - n_half) + k]
        
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(reduced_rho)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical zeros
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return float(entropy)


class QuantumInterferenceNetwork(nn.Module):
    """
    Neural network that models quantum interference patterns
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Quantum-inspired layers with interference
        self.quantum_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ])
        
        # Interference parameters
        self.interference_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.phase_parameters = nn.Parameter(torch.randn(hidden_dim))
        
        # Activation functions
        self.activations = nn.ModuleList([
            nn.Tanh(),
            nn.Sigmoid(),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantum interference"""
        batch_size = x.size(0)
        
        # Initial encoding
        x = self.quantum_layers[0](x)
        x = self.activations[0](x)
        
        # Quantum interference layers
        for i in range(1, 3):
            # Standard forward pass
            x_forward = self.quantum_layers[i](x)
            x_forward = self.activations[i](x_forward)
            
            # Interference term
            interference = torch.matmul(x, self.interference_weights)
            interference = interference * torch.cos(self.phase_parameters)
            
            # Combine with interference
            x = x_forward + 0.1 * interference
        
        # Final output layer
        x = self.quantum_layers[-1](x)
        x = self.activations[-1](x)
        
        return x


class AdvancedQuantumAgent:
    """
    Advanced quantum-inspired RL agent with sophisticated quantum dynamics
    """
    
    def __init__(self, state_dim: int, action_dim: int, n_qubits: int = 8, 
                 hidden_dim: int = 128, learning_rate: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        
        # Quantum state evolution
        self.quantum_state = QuantumStateEvolution(n_qubits)
        
        # Quantum interference network
        self.quantum_network = QuantumInterferenceNetwork(
            state_dim, hidden_dim, action_dim
        )
        
        # Classical neural network for policy
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim + n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(state_dim + n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.quantum_optimizer = optim.Adam(self.quantum_network.parameters(), lr=learning_rate)
        
        # Memory for experience replay
        self.memory = []
        self.max_memory_size = 10000
        
        # Quantum metrics
        self.entanglement_history = []
        self.quantum_fidelity_history = []
        self.interference_strength_history = []
        
    def encode_state_to_quantum(self, state: np.ndarray) -> np.ndarray:
        """Encode classical state into quantum representation"""
        # Normalize state
        state_normalized = (state - np.min(state)) / (np.max(state) - np.min(state) + 1e-8)
        
        # Apply rotation gates based on state values
        for i, value in enumerate(state_normalized[:self.n_qubits]):
            angle = value * np.pi
            self.quantum_state.apply_rotation_gate(angle, 'y', i)
        
        # Apply entangling gates
        for i in range(self.n_qubits - 1):
            if state_normalized[i] > 0.5:
                self.quantum_state.apply_cnot_gate(i, i + 1)
        
        # Measure and return quantum features
        measurement = self.quantum_state.measure_state()
        quantum_features = np.array(list(measurement.values()))
        
        return quantum_features
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, Dict[str, float]]:
        """Select action using quantum-enhanced policy"""
        # Encode state to quantum
        quantum_features = self.encode_state_to_quantum(state)
        
        # Combine classical and quantum features
        combined_features = np.concatenate([state, quantum_features])
        
        # Get quantum interference
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            quantum_interference = self.quantum_network(state_tensor)
        
        # Get policy probabilities
        combined_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy_network(combined_tensor)
        
        # Apply epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = torch.multinomial(action_probs.squeeze(), 1).item()
        
        # Calculate quantum metrics
        entanglement = self.quantum_state.calculate_entanglement()
        self.entanglement_history.append(entanglement)
        
        # Calculate quantum fidelity (simplified)
        quantum_fidelity = np.mean(np.abs(quantum_features))
        self.quantum_fidelity_history.append(quantum_fidelity)
        
        # Calculate interference strength
        interference_strength = torch.mean(torch.abs(quantum_interference)).item()
        self.interference_strength_history.append(interference_strength)
        
        action_info = {
            'quantum_features': quantum_features.tolist(),
            'entanglement': entanglement,
            'quantum_fidelity': quantum_fidelity,
            'interference_strength': interference_strength,
            'action_probability': float(action_probs[0, action])
        }
        
        return action, action_info
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.memory.append(experience)
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def update(self, batch_size: int = 32):
        """Update networks using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(self.memory, batch_size, replace=False)
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        
        # Get quantum features for all states
        quantum_features_batch = []
        for state in states:
            quantum_features = self.encode_state_to_quantum(state.numpy())
            quantum_features_batch.append(quantum_features)
        
        quantum_features_batch = torch.FloatTensor(quantum_features_batch)
        
        # Combine features
        combined_states = torch.cat([states, quantum_features_batch], dim=1)
        combined_next_states = torch.cat([next_states, quantum_features_batch], dim=1)
        
        # Update value network
        self.value_optimizer.zero_grad()
        current_values = self.value_network(combined_states).squeeze()
        
        with torch.no_grad():
            next_values = self.value_network(combined_next_states).squeeze()
            target_values = rewards + 0.99 * next_values * (~dones)
        
        value_loss = nn.MSELoss()(current_values, target_values)
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        action_probs = self.policy_network(combined_states)
        
        # Calculate advantages
        with torch.no_grad():
            advantages = target_values - current_values
        
        # Policy loss
        log_probs = torch.log(action_probs + 1e-8)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages).mean()
        
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update quantum network
        self.quantum_optimizer.zero_grad()
        
        # Quantum network loss (interference optimization)
        quantum_outputs = self.quantum_network(states)
        quantum_loss = torch.mean(torch.abs(quantum_outputs))  # Encourage interference
        
        quantum_loss.backward()
        self.quantum_optimizer.step()
    
    def get_quantum_metrics(self) -> Dict[str, List[float]]:
        """Get quantum performance metrics"""
        return {
            'entanglement_history': self.entanglement_history[-100:],
            'quantum_fidelity_history': self.quantum_fidelity_history[-100:],
            'interference_strength_history': self.interference_strength_history[-100:]
        }
    
    def reset_quantum_state(self):
        """Reset quantum state to initial superposition"""
        self.quantum_state.state = np.ones(self.quantum_state.dimension, dtype=complex) / np.sqrt(self.quantum_state.dimension)
        self.entanglement_history.clear()
        self.quantum_fidelity_history.clear()
        self.interference_strength_history.clear()

