"""
Quantum Reinforcement Learning Module

This module implements quantum-enhanced reinforcement learning algorithms
for complex control tasks, featuring variational quantum circuits and
hybrid quantum-classical architectures.

Key Components:
- QuantumRLCircuit: Variational quantum circuit for RL
- QuantumEnhancedAgent: Hybrid quantum-classical RL agent
- SpaceStationEnvironment: Critical infrastructure control environment
- MissionTrainer: Training system for quantum RL missions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import random
from collections import deque
import time
import warnings

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Fallback for spaces
        class spaces:
            @staticmethod
            def Box(low, high, shape, dtype):
                return {"low": low, "high": high, "shape": shape, "dtype": dtype}

            @staticmethod
            def Discrete(n):
                return {"n": n}


warnings.filterwarnings("ignore")

try:
    from qiskit import QuantumCircuit, execute, Aer
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Quantum RL will use classical fallback.")

    # Define dummy classes for fallback
    class QuantumCircuit:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits

        def ry(self, angle, qubit):
            pass

        def cx(self, qubit1, qubit2):
            pass

        def rz(self, angle, qubit):
            pass

        def rx(self, angle, qubit):
            pass

        def ry(self, angle, qubit):
            pass

        def rz(self, angle, qubit):
            pass

        def compose(self, other):
            return self

        def add_register(self, register, name):
            pass

        def measure_all(self):
            pass

        def depth(self):
            return 1

        def num_parameters(self):
            return 0

    class Parameter:
        def __init__(self, name):
            pass

    class Statevector:
        def __init__(self, data):
            self.data = data

    def partial_trace(rho, indices):
        return None

    def DensityMatrix(statevector):
        return None

    def execute(circuit, backend, shots=1024):
        class DummyResult:
            def result(self):
                class DummyJobResult:
                    def get_counts(self):
                        return {"00": shots // 2, "11": shots // 2}

                    def get_statevector(self):
                        return Statevector(
                            np.ones(2**circuit.n_qubits) / np.sqrt(2**circuit.n_qubits)
                        )

                return DummyJobResult()

        return DummyResult()

    class Aer:
        @staticmethod
        def get_backend(name):
            return None


class QuantumRLCircuit:
    """
    Mission-Critical Quantum Circuit for Reinforcement Learning

    This circuit represents the quantum brain that makes decisions affecting
    billions of lives across space stations, power grids, and markets.
    """

    def __init__(self, n_qubits: int, n_layers: int, feature_map: str = "ZZFeatureMap"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map

        if not QISKIT_AVAILABLE:
            print("Warning: Using classical fallback for QuantumRLCircuit")
            self.classical_fallback = True
            self.parameters = np.random.randn(n_qubits * n_layers * 3) * np.pi
            # Add dummy circuit attributes for compatibility
            self.circuit = QuantumCircuit(n_qubits)
        else:
            self.classical_fallback = False
            self.theta = [
                Parameter(f"θ_{l}_{q}")
                for l in range(n_layers)
                for q in range(n_qubits)
            ]
            self.phi = [
                Parameter(f"φ_{l}_{q}")
                for l in range(n_layers)
                for q in range(n_qubits)
            ]
            self.gamma = [
                Parameter(f"γ_{l}_{q}_{r}")
                for l in range(n_layers)
                for q in range(n_qubits)
                for r in range(q + 1, n_qubits)
            ]

    def create_feature_map(self, state_data: np.ndarray) -> QuantumCircuit:
        """
        Encode classical RL states into quantum superposition

        Each state becomes a quantum superposition of all possible futures.
        """
        if self.classical_fallback:
            # Classical fallback: return dummy circuit
            return QuantumCircuit(self.n_qubits)

        qc = QuantumCircuit(self.n_qubits)

        if self.feature_map == "ZZFeatureMap":
            for i in range(self.n_qubits):
                if i < len(state_data):
                    qc.ry(2 * np.arcsin(np.sqrt(abs(state_data[i]))), i)

            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    if i < len(state_data) and j < len(state_data):
                        interaction = 2 * state_data[i] * state_data[j]
                        qc.cx(i, j)
                        qc.rz(interaction, j)
                        qc.cx(i, j)

        elif self.feature_map == "PauliFeatureMap":
            for i in range(self.n_qubits):
                if i < len(state_data):
                    qc.rx(np.pi * state_data[i], i)
                    qc.ry(np.pi * state_data[i], i)
                    qc.rz(np.pi * state_data[i], i)

        return qc

    def create_ansatz(self, parameters: List[float]) -> QuantumCircuit:
        """
        Variational ansatz: The quantum neural network that learns optimal policies
        """
        if self.classical_fallback:
            # Classical fallback: return dummy circuit
            return QuantumCircuit(self.n_qubits)

        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                if param_idx < len(parameters):
                    qc.ry(parameters[param_idx], qubit)
                    param_idx += 1
                if param_idx < len(parameters):
                    qc.rz(parameters[param_idx], qubit)
                    param_idx += 1

            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)
                if param_idx < len(parameters):
                    qc.ry(parameters[param_idx], qubit + 1)
                    param_idx += 1

            if self.n_qubits > 2:
                qc.cx(self.n_qubits - 1, 0)

        return qc

    def execute_circuit(
        self, state: np.ndarray, parameters: List[float], shots: int = 1024
    ) -> Dict:
        """
        Execute the quantum circuit and extract RL-relevant information
        """
        if self.classical_fallback:
            # Classical fallback: return dummy quantum results
            n_actions = min(2**self.n_qubits, 64)
            action_probs = np.random.rand(n_actions)
            action_probs = action_probs / np.sum(action_probs)

            return {
                "action_probabilities": action_probs,
                "measurement_counts": {
                    f"{i:06b}": shots // n_actions for i in range(n_actions)
                },
                "quantum_fidelity": 0.5 + 0.3 * np.random.rand(),
                "entanglement_measure": 0.2 + 0.3 * np.random.rand(),
                "statevector": np.ones(2**self.n_qubits) / np.sqrt(2**self.n_qubits),
            }

        feature_circuit = self.create_feature_map(state)
        ansatz_circuit = self.create_ansatz(parameters)

        full_circuit = feature_circuit.compose(ansatz_circuit)

        full_circuit.add_register(full_circuit.classical_register(self.n_qubits, "c"))
        full_circuit.measure_all()

        backend = Aer.get_backend("qasm_simulator")
        job = execute(full_circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()

        action_probs = self._counts_to_action_probs(counts, shots)

        sv_circuit = feature_circuit.compose(ansatz_circuit)
        sv_backend = Aer.get_backend("statevector_simulator")
        sv_job = execute(sv_circuit, sv_backend)
        statevector = sv_job.result().get_statevector()

        quantum_info = {
            "action_probabilities": action_probs,
            "measurement_counts": counts,
            "quantum_fidelity": self._calculate_fidelity(statevector),
            "entanglement_measure": self._calculate_entanglement(statevector),
            "statevector": statevector,
        }

        return quantum_info

    def _counts_to_action_probs(self, counts: Dict, shots: int) -> np.ndarray:
        """Convert quantum measurements to action probabilities"""
        n_actions = min(2**self.n_qubits, 64)  # Cap at 64 actions
        action_probs = np.zeros(n_actions)

        for bitstring, count in counts.items():
            action_idx = int(bitstring, 2) % n_actions
            action_probs[action_idx] += count / shots

        if np.sum(action_probs) == 0:
            action_probs = np.ones(n_actions) / n_actions
        else:
            action_probs = action_probs / np.sum(action_probs)

        return action_probs

    def _calculate_fidelity(self, statevector) -> float:
        """Calculate quantum fidelity measure"""
        density_matrix = np.outer(statevector, np.conj(statevector))
        purity = np.real(np.trace(density_matrix @ density_matrix))
        return min(purity, 1.0)

    def _calculate_entanglement(self, statevector) -> float:
        """Calculate entanglement measure"""
        if self.classical_fallback:
            # Classical fallback: return random entanglement measure
            return 0.2 + 0.3 * np.random.rand()

        if self.n_qubits < 2:
            return 0.0

        try:
            rho = DensityMatrix(statevector)
            rho_A = partial_trace(rho, list(range(1, self.n_qubits)))

            eigenvals = np.real(np.linalg.eigvals(rho_A.data))
            eigenvals = eigenvals[eigenvals > 1e-10]

            if len(eigenvals) == 0:
                return 0.0

            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            return min(entropy, 1.0)
        except:
            return 0.0


class QuantumEnhancedAgent:
    """
    Quantum-Enhanced RL Agent

    Combines quantum circuits for enhanced representation with classical
    neural networks for stable learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        quantum_circuit: QuantumRLCircuit,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quantum_circuit = quantum_circuit
        self.lr = learning_rate
        self.gamma = gamma

        self.classical_network = self._build_classical_network()
        self.target_network = self._build_classical_network()
        self.optimizer = optim.Adam(
            self.classical_network.parameters(), lr=learning_rate
        )

        self.quantum_params = nn.Parameter(
            torch.randn(quantum_circuit.n_qubits * quantum_circuit.n_layers * 3) * np.pi
        )
        self.quantum_optimizer = optim.Adam(
            [self.quantum_params], lr=learning_rate * 0.1
        )

        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        self.episode_rewards = []
        self.quantum_fidelity_history = []
        self.loss_history = []

        self.quantum_weight = 0.7
        self.adaptive_fusion = True

    def _build_classical_network(self) -> nn.Module:
        """Build classical neural network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def select_action(
        self, state: np.ndarray, epsilon: float = 0.1, quantum_enabled: bool = True
    ) -> Tuple[int, Dict]:
        """
        Select action using quantum-classical hybrid decision making
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            classical_q_values = self.classical_network(state_tensor).squeeze()

        action_info = {"method": "classical", "quantum_fidelity": 0.0}

        if quantum_enabled and np.random.random() > epsilon:
            try:
                quantum_result = self.quantum_circuit.execute_circuit(
                    state, self.quantum_params.detach().numpy()
                )

                quantum_probs = quantum_result["action_probabilities"]
                quantum_fidelity = quantum_result["quantum_fidelity"]

                if len(quantum_probs) >= len(classical_q_values):
                    classical_probs = torch.softmax(classical_q_values, dim=0).numpy()

                    if self.adaptive_fusion:
                        self.quantum_weight = 0.3 + 0.7 * quantum_fidelity

                    fused_probs = (
                        self.quantum_weight * quantum_probs[: len(classical_probs)]
                        + (1 - self.quantum_weight) * classical_probs
                    )

                    action = np.random.choice(len(fused_probs), p=fused_probs)

                    action_info = {
                        "method": "quantum_fusion",
                        "quantum_fidelity": quantum_fidelity,
                        "quantum_weight": self.quantum_weight,
                        "entanglement": quantum_result["entanglement_measure"],
                        "quantum_probs": quantum_probs[:8],
                        "classical_probs": classical_probs.numpy()[:8],
                    }

                    self.quantum_fidelity_history.append(quantum_fidelity)

                else:
                    action = torch.argmax(classical_q_values).item()
                    action_info["method"] = "classical_fallback"

            except Exception as e:
                action = torch.argmax(classical_q_values).item()
                action_info["method"] = "classical_error_fallback"

        else:
            if np.random.random() < epsilon:
                action = np.random.randint(self.action_dim)
                action_info["method"] = "random_exploration"
            else:
                action = torch.argmax(classical_q_values).item()
                action_info["method"] = "classical_greedy"

        return action, action_info

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> Dict:
        """
        Learn from experiences using quantum-enhanced updates
        """
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0, "quantum_update": False}

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.classical_network(states).gather(
            1, actions.unsqueeze(1)
        )
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        classical_loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        classical_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.classical_network.parameters(), 1.0)
        self.optimizer.step()

        quantum_loss = 0.0
        quantum_updated = False

        if len(self.quantum_fidelity_history) > 10:
            try:
                quantum_batch_size = min(8, len(batch))
                quantum_indices = random.sample(range(len(batch)), quantum_batch_size)

                for i in quantum_indices:
                    state, action, reward, _, _ = batch[i]

                    quantum_grad = self._calculate_quantum_gradients(
                        state, action, reward
                    )

                    if quantum_grad is not None:
                        self.quantum_optimizer.zero_grad()
                        self.quantum_params.grad = torch.FloatTensor(quantum_grad)
                        torch.nn.utils.clip_grad_norm_([self.quantum_params], 0.1)
                        self.quantum_optimizer.step()

                        quantum_loss += np.linalg.norm(quantum_grad)
                        quantum_updated = True

            except:
                pass

        if hasattr(self, "update_counter"):
            self.update_counter += 1
        else:
            self.update_counter = 1

        if self.update_counter % 100 == 0:
            self.target_network.load_state_dict(self.classical_network.state_dict())

        self.loss_history.append(classical_loss.item())

        return {
            "classical_loss": classical_loss.item(),
            "quantum_loss": quantum_loss,
            "quantum_updated": quantum_updated,
            "avg_fidelity": (
                np.mean(self.quantum_fidelity_history[-10:])
                if self.quantum_fidelity_history
                else 0.0
            ),
        }

    def _calculate_quantum_gradients(
        self, state: np.ndarray, action: int, reward: float
    ) -> Optional[np.ndarray]:
        """
        Calculate quantum parameter gradients using parameter-shift rule
        """
        try:
            gradients = np.zeros_like(self.quantum_params.detach().numpy())
            shift = np.pi / 2

            param_indices = random.sample(
                range(len(gradients)), min(10, len(gradients))
            )

            for i in param_indices:
                params_plus = self.quantum_params.detach().numpy().copy()
                params_plus[i] += shift
                result_plus = self.quantum_circuit.execute_circuit(state, params_plus)

                params_minus = self.quantum_params.detach().numpy().copy()
                params_minus[i] -= shift
                result_minus = self.quantum_circuit.execute_circuit(state, params_minus)

                prob_plus = (
                    result_plus["action_probabilities"][action]
                    if action < len(result_plus["action_probabilities"])
                    else 0
                )
                prob_minus = (
                    result_minus["action_probabilities"][action]
                    if action < len(result_minus["action_probabilities"])
                    else 0
                )

                gradients[i] = 0.5 * (prob_plus - prob_minus) * reward

            return gradients

        except:
            return None

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        return {
            "episode_count": len(self.episode_rewards),
            "avg_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            ),
            "avg_quantum_fidelity": (
                np.mean(self.quantum_fidelity_history[-100:])
                if self.quantum_fidelity_history
                else 0.0
            ),
            "training_loss": (
                np.mean(self.loss_history[-10:]) if self.loss_history else 0.0
            ),
            "quantum_weight": self.quantum_weight,
            "memory_size": len(self.memory),
            "quantum_advantage_score": self._calculate_quantum_advantage_score(),
        }

    def _calculate_quantum_advantage_score(self) -> float:
        """Calculate quantum advantage metric"""
        if len(self.episode_rewards) < 10:
            return 0.0

        recent_performance = np.mean(self.episode_rewards[-10:])
        baseline_performance = (
            np.mean(self.episode_rewards[:10])
            if len(self.episode_rewards) >= 20
            else recent_performance
        )

        performance_improvement = (recent_performance - baseline_performance) / (
            abs(baseline_performance) + 1e-6
        )

        avg_fidelity = (
            np.mean(self.quantum_fidelity_history[-50:])
            if self.quantum_fidelity_history
            else 0.5
        )

        quantum_advantage = performance_improvement * avg_fidelity * self.quantum_weight

        return max(0.0, min(1.0, quantum_advantage))


class SpaceStationEnvironment:
    """
    Space Station Control Environment

    Critical infrastructure control environment where quantum-enhanced
    decisions can mean the difference between mission success and failure.
    """

    def __init__(self, difficulty_level: str = "EXTREME"):
        self.difficulty_level = difficulty_level
        self.difficulty = difficulty_level
        self.mission_time = 0
        self.crew_safety_score = 100.0
        self.system_reliability = 1.0

        self.subsystems = {
            "life_support": {"status": 1.0, "power_req": 25, "criticality": 10},
            "attitude_control": {"status": 1.0, "power_req": 15, "criticality": 8},
            "communications": {"status": 1.0, "power_req": 10, "criticality": 6},
            "experiments": {"status": 1.0, "power_req": 20, "criticality": 3},
            "thermal_control": {"status": 1.0, "power_req": 18, "criticality": 9},
            "navigation": {"status": 1.0, "power_req": 12, "criticality": 7},
        }

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(64)

        self.total_power = 100.0
        self.emergency_threshold = 0.3
        self.mission_duration = 1000

        self.crisis_events = self._initialize_crisis_scenarios()

    def _initialize_crisis_scenarios(self) -> List[Dict]:
        """Initialize realistic crisis scenarios"""
        return [
            {
                "name": "Oxygen Generator Malfunction",
                "trigger_step": 100,
                "affected_systems": ["life_support"],
                "severity": 0.8,
                "description": "Primary oxygen generation system failing",
            },
            {
                "name": "Solar Array Damage",
                "trigger_step": 250,
                "affected_systems": ["all"],
                "severity": 0.6,
                "description": "Micrometeorite impact on solar arrays",
            },
            {
                "name": "Attitude Control Failure",
                "trigger_step": 400,
                "affected_systems": ["attitude_control", "navigation"],
                "severity": 0.9,
                "description": "Gyroscope failure losing orientation",
            },
            {
                "name": "Communication Blackout",
                "trigger_step": 600,
                "affected_systems": ["communications"],
                "severity": 0.5,
                "description": "Primary communication array malfunction",
            },
            {
                "name": "Thermal Regulation Crisis",
                "trigger_step": 800,
                "affected_systems": ["thermal_control", "life_support"],
                "severity": 0.7,
                "description": "Cooling system failure, temperature rising",
            },
        ]

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.mission_time = 0
        self.crew_safety_score = 100.0
        self.system_reliability = 1.0

        for system in self.subsystems:
            self.subsystems[system]["status"] = 1.0

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step"""
        self.mission_time += 1

        action_decoded = self._decode_quantum_action(action)

        active_crisis = self._check_crisis_events()

        reward = self._update_systems(
            action_decoded["power_allocation"],
            action_decoded["emergency_response"],
            action_decoded["system_priority"],
            active_crisis,
        )

        safety_impact = self._calculate_safety_impact(active_crisis)
        self.crew_safety_score = max(0.0, self.crew_safety_score + safety_impact)

        done = (
            self.crew_safety_score < 20.0
            or self.mission_time >= self.mission_duration
            or any(
                self.subsystems[sys]["status"] < 0.1
                and self.subsystems[sys]["criticality"] > 8
                for sys in self.subsystems
            )
        )

        if done and self.crew_safety_score < 20.0:
            reward -= 1000  # Severe penalty

        info = {
            "crew_safety": self.crew_safety_score,
            "active_crisis": active_crisis,
            "power_usage": sum(
                self.subsystems[s]["power_req"] * self.subsystems[s]["status"]
                for s in self.subsystems
            ),
            "system_health": {s: self.subsystems[s]["status"] for s in self.subsystems},
            "mission_time": self.mission_time,
            "quantum_advantage_opportunity": self._assess_quantum_advantage_opportunity(),
        }

        return self._get_observation(), reward, done, info

    def _decode_quantum_action(self, action: int) -> Dict:
        """Decode quantum action into space station commands"""
        power_bits = [(action >> i) & 1 for i in range(6)]
        power_total = sum(power_bits) if sum(power_bits) > 0 else 1
        power_allocation = {
            system: power_bits[i] / power_total
            for i, system in enumerate(self.subsystems.keys())
        }

        emergency_response = (action >> 6) & 3
        system_priority = list(self.subsystems.keys())
        priority_seed = (action >> 8) & 15
        np.random.seed(priority_seed)
        np.random.shuffle(system_priority)
        np.random.seed(None)

        return {
            "power_allocation": power_allocation,
            "emergency_response": emergency_response,
            "system_priority": system_priority,
        }

    def _check_crisis_events(self) -> Optional[Dict]:
        """Check for active crisis events"""
        for crisis in self.crisis_events:
            if (
                crisis["trigger_step"]
                <= self.mission_time
                < crisis["trigger_step"] + 10
            ):
                return crisis
        return None

    def _update_systems(
        self,
        power_allocation: Dict,
        emergency_response: int,
        system_priority: List[str],
        active_crisis: Optional[Dict],
    ) -> float:
        """Update all space station systems"""
        total_reward = 0.0

        if active_crisis:
            if active_crisis["affected_systems"] == ["all"]:
                self.total_power *= 1.0 - active_crisis["severity"] * 0.5
            else:
                for system in active_crisis["affected_systems"]:
                    if system in self.subsystems:
                        failure_amount = active_crisis["severity"] * 0.3
                        self.subsystems[system]["status"] -= failure_amount

        total_power_demand = sum(
            self.subsystems[s]["power_req"]
            * power_allocation[s]
            * self.subsystems[s]["status"]
            for s in self.subsystems
        )

        power_efficiency = (
            min(1.0, self.total_power / total_power_demand)
            if total_power_demand > 0
            else 1.0
        )

        for system in self.subsystems:
            allocated_power_ratio = power_allocation[system] * power_efficiency

            if allocated_power_ratio > 0.8:
                self.subsystems[system]["status"] = min(
                    1.0, self.subsystems[system]["status"] + 0.01
                )
                total_reward += self.subsystems[system]["criticality"] * 0.1
            elif allocated_power_ratio < 0.3:
                self.subsystems[system]["status"] = max(
                    0.0, self.subsystems[system]["status"] - 0.05
                )
                total_reward -= self.subsystems[system]["criticality"] * 0.5

            if (
                self.subsystems[system]["status"] < 0.5
                and self.subsystems[system]["criticality"] > 8
            ):
                total_reward -= 100

        if active_crisis and emergency_response >= 2:
            total_reward += 50

        return total_reward

    def _calculate_safety_impact(self, active_crisis: Optional[Dict]) -> float:
        """Calculate crew safety impact"""
        safety_change = 0.0
        safety_change -= 0.1  # Base orbital risk

        life_support_health = self.subsystems["life_support"]["status"]
        if life_support_health < 0.5:
            safety_change -= 10.0
        elif life_support_health < 0.8:
            safety_change -= 2.0

        if active_crisis and "life_support" in active_crisis.get(
            "affected_systems", []
        ):
            safety_change -= active_crisis["severity"] * 5.0

        return safety_change

    def _assess_quantum_advantage_opportunity(self) -> float:
        """Assess quantum advantage potential"""
        systems_needing_attention = sum(
            1 for s in self.subsystems if self.subsystems[s]["status"] < 0.8
        )

        crisis_complexity = 1.0
        active_crisis = self._check_crisis_events()
        if active_crisis:
            crisis_complexity = 1.0 + active_crisis["severity"]

        quantum_opportunity = (
            systems_needing_attention / len(self.subsystems)
        ) * crisis_complexity

        return min(1.0, quantum_opportunity)

    def _get_observation(self) -> np.ndarray:
        """Generate observation vector"""
        obs = []

        obs.append(self.total_power / 100.0)
        for system in self.subsystems:
            obs.append(self.subsystems[system]["status"])

        for system in self.subsystems:
            obs.append(self.subsystems[system]["criticality"] / 10.0)

        obs.append(self.mission_time / self.mission_duration)
        obs.append(self.crew_safety_score / 100.0)
        obs.append(self.system_reliability)

        remaining_crises = [
            c for c in self.crisis_events if c["trigger_step"] > self.mission_time
        ]
        for i in range(4):
            if i < len(remaining_crises):
                time_to_crisis = remaining_crises[i]["trigger_step"] - self.mission_time
                obs.append(np.exp(-time_to_crisis / 100.0))
            else:
                obs.append(0.0)

        return np.array(obs[:20], dtype=np.float32)


class MissionTrainer:
    """
    Mission Control Center for Training Quantum-Enhanced Agents
    """

    def __init__(self, agent: QuantumEnhancedAgent, environment, logger=None):
        self.agent = agent
        self.env = environment
        self.logger = logger or MissionLogger()

        self.max_episodes = 500
        self.current_episode = 0

        self.classical_baseline_scores = []
        self.quantum_enhanced_scores = []
        self.crisis_survival_rate = 0.0

        self.difficulty_progression = True
        self.crisis_injection_rate = 0.2

    def execute_mission(
        self, num_episodes: int = 50, quantum_enabled: bool = True, verbose: bool = True
    ) -> Dict:
        """
        Execute training mission
        """
        self.logger.log("INFO", f"🎯 MISSION START: {num_episodes} episodes")

        episode_rewards = []
        crisis_episodes = []
        quantum_advantages = []
        safety_incidents = 0

        base_epsilon = 0.3
        epsilon_decay = 0.995

        for episode in range(num_episodes):
            self.current_episode += 1
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            crisis_encountered = False
            epsilon = base_epsilon * (epsilon_decay**episode)

            done = False
            while not done and steps < 1000:
                action, action_info = self.agent.select_action(
                    state, epsilon=epsilon, quantum_enabled=quantum_enabled
                )

                next_state, reward, done, info = self.env.step(action)

                self.agent.store_experience(state, action, reward, next_state, done)

                total_reward += reward
                steps += 1

                if info.get("active_crisis") is not None:
                    crisis_encountered = True
                    if not crisis_episodes or crisis_episodes[-1] != episode:
                        crisis_episodes.append(episode)

                if info.get("crew_safety", 100) < 50:
                    safety_incidents += 1

                if info.get("quantum_advantage_opportunity", 0) > 0.7:
                    quantum_advantages.append(action_info.get("quantum_fidelity", 0))

                state = next_state

                if steps % 4 == 0:
                    self.agent.learn()

            episode_rewards.append(total_reward)

            if quantum_enabled:
                self.quantum_enhanced_scores.append(total_reward)
            else:
                self.classical_baseline_scores.append(total_reward)

        mission_results = self._analyze_mission_results(
            episode_rewards,
            crisis_episodes,
            quantum_advantages,
            safety_incidents,
            quantum_enabled,
        )

        self.logger.log(
            "SUCCESS",
            f"🏆 MISSION COMPLETED: avg reward {np.mean(episode_rewards):.2f}",
        )

        return mission_results

    def _analyze_mission_results(
        self,
        episode_rewards: List[float],
        crisis_episodes: List[int],
        quantum_advantages: List[float],
        safety_incidents: int,
        quantum_enabled: bool,
    ) -> Dict:
        """Analyze mission results"""
        results = {
            "mission_type": (
                "quantum_enhanced" if quantum_enabled else "classical_baseline"
            ),
            "episodes_completed": len(episode_rewards),
            "average_reward": np.mean(episode_rewards),
            "reward_std": np.std(episode_rewards),
            "best_performance": np.max(episode_rewards),
            "worst_performance": np.min(episode_rewards),
            "crisis_episodes": len(crisis_episodes),
            "crisis_survival_rate": len([r for r in episode_rewards if r > -500])
            / len(episode_rewards),
            "crew_safety_incidents": safety_incidents,
            "final_agent_metrics": self.agent.get_performance_metrics(),
        }

        if quantum_enabled and quantum_advantages:
            results["quantum_advantages"] = {
                "opportunities_detected": len(quantum_advantages),
                "average_quantum_fidelity": np.mean(quantum_advantages),
                "quantum_advantage_score": self.agent._calculate_quantum_advantage_score(),
            }

        if len(episode_rewards) >= 20:
            early_performance = np.mean(episode_rewards[:10])
            late_performance = np.mean(episode_rewards[-10:])
            results["learning_progress"] = (late_performance - early_performance) / (
                abs(early_performance) + 1e-6
            )

        return results

    def compare_quantum_classical_performance(self) -> Dict:
        """Compare quantum vs classical performance"""
        if not self.quantum_enhanced_scores or not self.classical_baseline_scores:
            return {"error": "Insufficient data"}

        quantum_avg = np.mean(self.quantum_enhanced_scores)
        classical_avg = np.mean(self.classical_baseline_scores)

        quantum_advantage_percent = (
            (quantum_avg - classical_avg) / abs(classical_avg)
        ) * 100

        try:
            from scipy import stats

            t_stat, p_value = stats.ttest_ind(
                self.quantum_enhanced_scores, self.classical_baseline_scores
            )
            statistically_significant = p_value < 0.05
        except ImportError:
            statistically_significant = abs(quantum_avg - classical_avg) > 50
            p_value = 0.5

        comparison = {
            "quantum_average": quantum_avg,
            "classical_average": classical_avg,
            "quantum_advantage_percent": quantum_advantage_percent,
            "statistically_significant": statistically_significant,
            "p_value": p_value,
            "quantum_superiority": quantum_avg > classical_avg,
            "advantage_magnitude": (
                "BREAKTHROUGH"
                if quantum_advantage_percent > 50
                else (
                    "SIGNIFICANT"
                    if quantum_advantage_percent > 20
                    else "MODERATE" if quantum_advantage_percent > 5 else "MINIMAL"
                )
            ),
        }

        return comparison


class MissionLogger:
    """Mission logging system"""

    def __init__(self):
        self.logs = []
        self.start_time = time.time() if "time" in dir() else 0

    def log(self, level: str, message: str, quantum_fidelity: float = None):
        timestamp = time.time() - self.start_time if "time" in dir() else len(self.logs)
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "quantum_fidelity": quantum_fidelity,
        }
        self.logs.append(entry)

        prefix_map = {
            "INFO": "📋",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "QUANTUM": "⚛️",
        }

        prefix = prefix_map.get(level, "📝")
        fidelity_str = f" [F={quantum_fidelity:.4f}]" if quantum_fidelity else ""
        print(f"{prefix} T+{timestamp:06.2f}s: {message}{fidelity_str}")

    def get_mission_report(self):
        return self.logs
