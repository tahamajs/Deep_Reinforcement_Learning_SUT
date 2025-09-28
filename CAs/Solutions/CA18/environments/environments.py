import torch
import torch.nn as nn
import numpy as np
import gym
from gym import spaces
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from collections import deque
import copy
import random
import networkx as nx
import time

# Quantum-Inspired Environment
class QuantumEnvironment(gym.Env):
    """Quantum-inspired reinforcement learning environment"""

    def __init__(
        self,
        n_qubits: int = 4,
        max_steps: int = 100,
        noise_level: float = 0.1,
        reward_type: str = "fidelity",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.max_steps = max_steps
        self.noise_level = noise_level
        self.reward_type = reward_type

        # Action space: quantum gates (rotation angles for each qubit)
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(n_qubits * 3,), dtype=np.float32
        )  # RX, RY, RZ for each qubit

        # Observation space: quantum state amplitudes (complex)
        state_dim = 2 ** n_qubits * 2  # Real and imaginary parts
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(state_dim,), dtype=np.float32
        )

        # Target quantum state (Bell state or GHZ state)
        self.target_state = self._create_target_state()

        self.reset()

    def _create_target_state(self) -> np.ndarray:
        """Create target quantum state"""
        if self.n_qubits == 2:
            # Bell state: (|00> + |11>)/sqrt(2)
            state = np.zeros(4, dtype=np.complex64)
            state[0] = 1/np.sqrt(2)  # |00>
            state[3] = 1/np.sqrt(2)  # |11>
        else:
            # GHZ state for more qubits
            state = np.zeros(2**self.n_qubits, dtype=np.complex64)
            state[0] = 1/np.sqrt(2)  # |00...0>
            state[-1] = 1/np.sqrt(2)  # |11...1>

        return state

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_state = np.zeros(2**self.n_qubits, dtype=np.complex64)
        self.current_state[0] = 1.0  # Start in |00...0> state
        self.steps = 0
        self.done = False

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute quantum gate action"""

        # Apply quantum gates
        self._apply_quantum_gates(action)

        # Add noise
        if self.noise_level > 0:
            self._add_quantum_noise()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        self.steps += 1
        self.done = self.steps >= self.max_steps

        # Additional info
        info = {
            'fidelity': self._compute_fidelity(),
            'purity': self._compute_purity(),
            'entanglement': self._compute_entanglement(),
            'steps': self.steps,
        }

        return self._get_observation(), reward, self.done, info

    def _apply_quantum_gates(self, action: np.ndarray):
        """Apply quantum rotation gates"""
        # Reshape action into RX, RY, RZ angles for each qubit
        angles = action.reshape(self.n_qubits, 3)  # [n_qubits, 3]

        for qubit in range(self.n_qubits):
            rx_angle, ry_angle, rz_angle = angles[qubit]

            # Apply RX gate
            self._apply_single_qubit_gate(qubit, self._rx_gate(rx_angle))

            # Apply RY gate
            self._apply_single_qubit_gate(qubit, self._ry_gate(ry_angle))

            # Apply RZ gate
            self._apply_single_qubit_gate(qubit, self._rz_gate(rz_angle))

    def _apply_single_qubit_gate(self, qubit: int, gate: np.ndarray):
        """Apply single qubit gate"""
        # Tensor product with identity on other qubits
        full_gate = np.array([[1]], dtype=np.complex64)

        for q in range(self.n_qubits):
            if q == qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2, dtype=np.complex64))

        # Apply gate to state
        self.current_state = full_gate @ self.current_state

    def _rx_gate(self, angle: float) -> np.ndarray:
        """RX rotation gate"""
        c, s = np.cos(angle/2), np.sin(angle/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex64)

    def _ry_gate(self, angle: float) -> np.ndarray:
        """RY rotation gate"""
        c, s = np.cos(angle/2), np.sin(angle/2)
        return np.array([[c, -s], [s, c]], dtype=np.complex64)

    def _rz_gate(self, angle: float) -> np.ndarray:
        """RZ rotation gate"""
        return np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]], dtype=np.complex64)

    def _add_quantum_noise(self):
        """Add quantum noise (decoherence)"""
        # Simple amplitude damping
        damping_factor = np.exp(-self.noise_level * self.steps / self.max_steps)
        self.current_state *= damping_factor

        # Add phase noise
        phase_noise = np.random.normal(0, self.noise_level, len(self.current_state))
        self.current_state *= np.exp(1j * phase_noise)

        # Renormalize
        norm = np.linalg.norm(self.current_state)
        if norm > 0:
            self.current_state /= norm

    def _compute_reward(self) -> float:
        """Compute reward based on quantum state quality"""
        if self.reward_type == "fidelity":
            fidelity = self._compute_fidelity()
            return fidelity * 10  # Scale reward
        elif self.reward_type == "purity":
            purity = self._compute_purity()
            return purity * 5
        elif self.reward_type == "entanglement":
            entanglement = self._compute_entanglement()
            return entanglement * 2
        else:
            return 0.0

    def _compute_fidelity(self) -> float:
        """Compute fidelity with target state"""
        overlap = np.abs(np.vdot(self.current_state, self.target_state))**2
        return overlap

    def _compute_purity(self) -> float:
        """Compute state purity"""
        density_matrix = np.outer(self.current_state, self.current_state.conj())
        purity = np.real(np.trace(density_matrix @ density_matrix))
        return purity

    def _compute_entanglement(self) -> float:
        """Compute entanglement measure (simplified)"""
        # Use linear entropy as entanglement proxy
        purity = self._compute_purity()
        return 1 - purity

    def _get_observation(self) -> np.ndarray:
        """Get observation (real and imaginary parts of quantum state)"""
        return np.concatenate([self.current_state.real, self.current_state.imag]).astype(np.float32)

    def render(self, mode: str = 'human'):
        """Render quantum state"""
        if mode == 'human':
            print(f"Step {self.steps}/{self.max_steps}")
            print(f"Current state: {self.current_state}")
            print(f"Fidelity: {self._compute_fidelity():.4f}")
            print(f"Purity: {self._compute_purity():.4f}")
            print(f"Entanglement: {self._compute_entanglement():.4f}")
        elif mode == 'rgb_array':
            # Create simple visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot real parts
            ax1.bar(range(len(self.current_state)), self.current_state.real)
            ax1.set_title('Real Amplitudes')
            ax1.set_xlabel('Basis State')
            ax1.set_ylabel('Amplitude')

            # Plot imaginary parts
            ax2.bar(range(len(self.current_state)), self.current_state.imag)
            ax2.set_title('Imaginary Amplitudes')
            ax2.set_xlabel('Basis State')
            ax2.set_ylabel('Amplitude')

            plt.tight_layout()
            plt.close()

            # Convert to RGB array (simplified)
            return np.random.randint(0, 255, (100, 100, 3))  # Placeholder

# Causal Bandit Environment
class CausalBanditEnvironment(gym.Env):
    """Causal bandit environment with hidden causal structure"""

    def __init__(
        self,
        n_arms: int = 5,
        n_context_vars: int = 3,
        causal_hidden: bool = True,
        noise_level: float = 0.1,
    ):
        super().__init__()

        self.n_arms = n_arms
        self.n_context_vars = n_context_vars
        self.causal_hidden = causal_hidden
        self.noise_level = noise_level

        # Action space: choose arm
        self.action_space = spaces.Discrete(n_arms)

        # Observation space: context variables
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(n_context_vars,), dtype=np.float32
        )

        # Create causal graph
        self.causal_graph = self._create_causal_graph()

        # True reward function parameters
        self.arm_params = np.random.randn(n_arms, n_context_vars + 1) * 0.5

        self.reset()

    def _create_causal_graph(self) -> nx.DiGraph:
        """Create causal graph for the environment"""
        G = nx.DiGraph()

        # Add context variables
        for i in range(self.n_context_vars):
            G.add_node(f'X{i}')

        # Add arms
        for i in range(self.n_arms):
            G.add_node(f'A{i}')

        # Add reward node
        G.add_node('R')

        # Add causal edges (context -> arms -> reward)
        for i in range(self.n_context_vars):
            for j in range(self.n_arms):
                if np.random.random() < 0.3:  # 30% chance of causal link
                    G.add_edge(f'X{i}', f'A{j}')

        for i in range(self.n_arms):
            G.add_edge(f'A{i}', 'R')

        return G

    def reset(self) -> np.ndarray:
        """Reset environment"""
        # Sample context variables
        self.context = np.random.randn(self.n_context_vars).astype(np.float32)
        self.done = False

        return self.context.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action in causal bandit"""

        # Compute true reward based on causal structure
        reward = self._compute_causal_reward(action)

        # Add noise
        reward += np.random.normal(0, self.noise_level)

        # Always done after one step in bandit setting
        self.done = True

        info = {
            'true_reward': reward - np.random.normal(0, self.noise_level),
            'context': self.context.copy(),
            'causal_links': self._get_causal_links(action),
        }

        return self.context.copy(), reward, self.done, info

    def _compute_causal_reward(self, action: int) -> float:
        """Compute reward based on causal relationships"""
        # Base reward from arm parameters
        reward = self.arm_params[action, -1]  # Bias term

        # Add effects from context variables that causally affect this arm
        for i in range(self.n_context_vars):
            if self.causal_graph.has_edge(f'X{i}', f'A{action}'):
                reward += self.arm_params[action, i] * self.context[i]

        return reward

    def _get_causal_links(self, action: int) -> List[str]:
        """Get causal links for the chosen action"""
        links = []
        for i in range(self.n_context_vars):
            if self.causal_graph.has_edge(f'X{i}', f'A{action}'):
                links.append(f'X{i} -> A{action}')
        return links

    def render(self, mode: str = 'human'):
        """Render causal bandit state"""
        if mode == 'human':
            print(f"Context: {self.context}")
            print("Causal Graph:")
            for edge in self.causal_graph.edges():
                print(f"  {edge[0]} -> {edge[1]}")

# Multi-Agent Quantum Environment
class MultiAgentQuantumEnvironment(gym.Env):
    """Multi-agent quantum environment for cooperative/competitive quantum control"""

    def __init__(
        self,
        n_agents: int = 2,
        n_qubits_per_agent: int = 2,
        cooperation_bonus: float = 0.5,
        max_steps: int = 50,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_qubits_per_agent = n_qubits_per_agent
        self.total_qubits = n_agents * n_qubits_per_agent
        self.cooperation_bonus = cooperation_bonus
        self.max_steps = max_steps

        # Action space per agent: quantum gates for their qubits
        agent_action_dim = n_qubits_per_agent * 3  # RX, RY, RZ per qubit
        self.action_space = spaces.Tuple([
            spaces.Box(low=-np.pi, high=np.pi, shape=(agent_action_dim,), dtype=np.float32)
            for _ in range(n_agents)
        ])

        # Observation space per agent: their qubits' state + global entanglement measure
        agent_obs_dim = (2 ** n_qubits_per_agent * 2) + 1  # Local state + entanglement
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-1, high=1, shape=(agent_obs_dim,), dtype=np.float32)
            for _ in range(n_agents)
        ])

        # Global target state (maximally entangled state)
        self.target_state = self._create_ghz_state()

        self.reset()

    def _create_ghz_state(self) -> np.ndarray:
        """Create GHZ state for all qubits"""
        state = np.zeros(2 ** self.total_qubits, dtype=np.complex64)
        state[0] = 1/np.sqrt(2)  # |00...0>
        state[-1] = 1/np.sqrt(2)  # |11...1>
        return state

    def reset(self) -> List[np.ndarray]:
        """Reset multi-agent quantum environment"""
        self.global_state = np.zeros(2 ** self.total_qubits, dtype=np.complex64)
        self.global_state[0] = 1.0  # Start in |00...0>

        self.steps = 0
        self.done = False

        return self._get_observations()

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """Execute multi-agent quantum actions"""

        # Apply actions from all agents
        for agent_id, action in enumerate(actions):
            self._apply_agent_action(agent_id, action)

        # Compute rewards
        rewards = self._compute_multi_agent_rewards()

        # Check termination
        self.steps += 1
        self.done = self.steps >= self.max_steps

        observations = self._get_observations()

        info = {
            'global_fidelity': self._compute_global_fidelity(),
            'agent_contributions': self._compute_agent_contributions(),
            'entanglement_measure': self._compute_entanglement(),
            'steps': self.steps,
        }

        return observations, rewards, self.done, info

    def _apply_agent_action(self, agent_id: int, action: np.ndarray):
        """Apply action from specific agent"""
        start_qubit = agent_id * self.n_qubits_per_agent
        end_qubit = start_qubit + self.n_qubits_per_agent

        # Apply gates to agent's qubits
        angles = action.reshape(self.n_qubits_per_agent, 3)

        for i, qubit_idx in enumerate(range(start_qubit, end_qubit)):
            rx_angle, ry_angle, rz_angle = angles[i]

            # Apply RX
            self._apply_single_qubit_gate_global(qubit_idx, self._rx_gate(rx_angle))
            # Apply RY
            self._apply_single_qubit_gate_global(qubit_idx, self._ry_gate(ry_angle))
            # Apply RZ
            self._apply_single_qubit_gate_global(qubit_idx, self._rz_gate(rz_angle))

    def _apply_single_qubit_gate_global(self, qubit: int, gate: np.ndarray):
        """Apply single qubit gate to global state"""
        full_gate = np.array([[1]], dtype=np.complex64)

        for q in range(self.total_qubits):
            if q == qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2, dtype=np.complex64))

        self.global_state = full_gate @ self.global_state

    def _rx_gate(self, angle: float) -> np.ndarray:
        c, s = np.cos(angle/2), np.sin(angle/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex64)

    def _ry_gate(self, angle: float) -> np.ndarray:
        c, s = np.cos(angle/2), np.sin(angle/2)
        return np.array([[c, -s], [s, c]], dtype=np.complex64)

    def _rz_gate(self, angle: float) -> np.ndarray:
        return np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]], dtype=np.complex64)

    def _compute_multi_agent_rewards(self) -> List[float]:
        """Compute rewards for all agents"""
        global_fidelity = self._compute_global_fidelity()
        agent_contributions = self._compute_agent_contributions()

        rewards = []
        for agent_id in range(self.n_agents):
            # Individual reward based on contribution
            individual_reward = agent_contributions[agent_id] * global_fidelity * 10

            # Cooperation bonus
            cooperation_reward = self.cooperation_bonus * global_fidelity

            total_reward = individual_reward + cooperation_reward
            rewards.append(total_reward)

        return rewards

    def _compute_global_fidelity(self) -> float:
        """Compute fidelity with target GHZ state"""
        overlap = np.abs(np.vdot(self.global_state, self.target_state))**2
        return overlap

    def _compute_agent_contributions(self) -> List[float]:
        """Compute contribution of each agent to global state"""
        contributions = []

        for agent_id in range(self.n_agents):
            # Compute reduced density matrix for agent's qubits
            reduced_state = self._get_agent_reduced_state(agent_id)
            fidelity = np.abs(np.vdot(reduced_state, reduced_state))**2  # Purity measure
            contributions.append(fidelity)

        # Normalize contributions
        total = sum(contributions)
        if total > 0:
            contributions = [c / total for c in contributions]

        return contributions

    def _get_agent_reduced_state(self, agent_id: int) -> np.ndarray:
        """Get reduced state for agent's qubits"""
        # Simplified: return portion of global state corresponding to agent
        start_idx = agent_id * (2 ** self.n_qubits_per_agent)
        end_idx = start_idx + (2 ** self.n_qubits_per_agent)
        return self.global_state[start_idx:end_idx]

    def _compute_entanglement(self) -> float:
        """Compute global entanglement measure"""
        # Use linear entropy of reduced states as entanglement proxy
        total_entanglement = 0
        for agent_id in range(self.n_agents):
            reduced_state = self._get_agent_reduced_state(agent_id)
            purity = np.abs(np.vdot(reduced_state, reduced_state))**2
            total_entanglement += 1 - purity

        return total_entanglement / self.n_agents

    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents"""
        observations = []

        for agent_id in range(self.n_agents):
            # Local state
            local_state = self._get_agent_reduced_state(agent_id)
            local_obs = np.concatenate([local_state.real, local_state.imag])

            # Global entanglement measure
            entanglement = self._compute_entanglement()

            # Combine observations
            obs = np.concatenate([local_obs, [entanglement]]).astype(np.float32)
            observations.append(obs)

        return observations

    def render(self, mode: str = 'human'):
        """Render multi-agent quantum environment"""
        if mode == 'human':
            print(f"Step {self.steps}/{self.max_steps}")
            print(f"Global fidelity: {self._compute_global_fidelity():.4f}")
            print(f"Global entanglement: {self._compute_entanglement():.4f}")

            for agent_id in range(self.n_agents):
                contrib = self._compute_agent_contributions()[agent_id]
                print(f"Agent {agent_id} contribution: {contrib:.4f}")

# Federated Learning Environment
class FederatedLearningEnvironment(gym.Env):
    """Environment for federated reinforcement learning"""

    def __init__(
        self,
        n_clients: int = 5,
        n_rounds: int = 10,
        heterogeneity_level: float = 0.5,
        communication_cost: float = 0.1,
    ):
        super().__init__()

        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.heterogeneity_level = heterogeneity_level
        self.communication_cost = communication_cost

        # Action space: which clients to select for training
        self.action_space = spaces.MultiBinary(n_clients)

        # Observation space: client states and global model performance
        obs_dim = n_clients * 3 + 2  # client data sizes, losses, accuracies + global metrics
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)

        # Initialize client data distributions
        self.client_data_sizes = np.random.randint(100, 1000, n_clients)
        self.client_heterogeneity = np.random.uniform(0, heterogeneity_level, n_clients)

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset federated learning environment"""
        self.current_round = 0
        self.global_model_accuracy = 0.1  # Start with poor performance
        self.client_local_losses = np.random.uniform(0.5, 1.0, self.n_clients)
        self.client_local_accuracies = np.random.uniform(0.1, 0.3, self.n_clients)
        self.done = False

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute federated learning round"""

        selected_clients = np.where(action == 1)[0]

        if len(selected_clients) == 0:
            # No clients selected - penalty
            reward = -1.0
            communication_cost = 0
        else:
            # Perform federated aggregation
            reward, communication_cost = self._perform_federated_round(selected_clients)

        # Update round
        self.current_round += 1
        self.done = self.current_round >= self.n_rounds

        info = {
            'selected_clients': selected_clients.tolist(),
            'global_accuracy': self.global_model_accuracy,
            'communication_cost': communication_cost,
            'round': self.current_round,
        }

        return self._get_observation(), reward, self.done, info

    def _perform_federated_round(self, selected_clients: np.ndarray) -> Tuple[float, float]:
        """Perform one round of federated learning"""

        # Simulate local training
        local_updates = []
        total_data_size = 0

        for client_id in selected_clients:
            # Local training improvement (affected by heterogeneity)
            improvement = np.random.uniform(0.01, 0.05) * (1 - self.client_heterogeneity[client_id])
            self.client_local_losses[client_id] -= improvement
            self.client_local_accuracies[client_id] += improvement

            # Collect update
            update_size = self.client_data_sizes[client_id]
            local_updates.append(update_size)
            total_data_size += update_size

        # Aggregation (FedAvg-style)
        if len(local_updates) > 0:
            # Global model improvement based on participation
            participation_rate = len(selected_clients) / self.n_clients
            heterogeneity_penalty = np.mean([self.client_heterogeneity[cid] for cid in selected_clients])

            global_improvement = participation_rate * 0.1 * (1 - heterogeneity_penalty)
            self.global_model_accuracy = min(1.0, self.global_model_accuracy + global_improvement)

        # Communication cost
        communication_cost = self.communication_cost * len(selected_clients)

        # Reward: accuracy improvement minus costs
        accuracy_reward = self.global_model_accuracy * 10
        cost_penalty = communication_cost * 2
        reward = accuracy_reward - cost_penalty

        return reward, communication_cost

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Normalize data sizes
        normalized_sizes = self.client_data_sizes / np.max(self.client_data_sizes)

        observation = np.concatenate([
            normalized_sizes,  # Client data sizes
            self.client_local_losses,  # Local losses
            self.client_local_accuracies,  # Local accuracies
            [self.global_model_accuracy, self.current_round / self.n_rounds]  # Global metrics
        ]).astype(np.float32)

        return observation

    def render(self, mode: str = 'human'):
        """Render federated learning state"""
        if mode == 'human':
            print(f"Round {self.current_round}/{self.n_rounds}")
            print(f"Global model accuracy: {self.global_model_accuracy:.4f}")
            print(f"Average client loss: {np.mean(self.client_local_losses):.4f}")
            print(f"Average client accuracy: {np.mean(self.client_local_accuracies):.4f}")

print("✅ Advanced Environments implementations complete!")
print("Components implemented:")
print("- QuantumEnvironment: Quantum-inspired RL environment with gate operations")
print("- CausalBanditEnvironment: Causal bandit with hidden causal structure")
print("- MultiAgentQuantumEnvironment: Multi-agent quantum control environment")
print("- FederatedLearningEnvironment: Federated RL environment with client selection")