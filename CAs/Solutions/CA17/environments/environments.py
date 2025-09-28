"""
Environments Module

This module contains custom reinforcement learning environments
for various RL paradigms and experimental setups.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Tuple, Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx
import random
import math

# Base Environment Classes
class BaseEnvironment(gym.Env):
    """Base class for custom RL environments"""

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed(seed)
        self.reset()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'human'):
        """Render environment"""
        pass

    def close(self):
        """Clean up environment"""
        pass

# Continuous Control Environments
class ContinuousMountainCar(BaseEnvironment):
    """Continuous version of Mountain Car environment"""

    def __init__(self, goal_velocity: float = 0.0, seed: Optional[int] = None):
        super().__init__(seed)

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        position, velocity = self.state

        # Apply action
        force = min(max(action[0], -1.0), 1.0)
        velocity += force * self.power - 0.0025 * np.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)

        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        # Reset velocity if at boundaries
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Check termination
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = -1.0  # Default reward

        # Reward shaping
        if done:
            reward = 100.0
        elif position >= self.goal_position:
            reward = 10.0

        self.state = np.array([position, velocity])
        return self.state, reward, done, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return self.state, {}

    def _height(self, xs: np.ndarray) -> np.ndarray:
        """Height function for mountain car"""
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode: str = 'human'):
        """Render environment"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(8, 4))
            self.ax.set_xlim(self.min_position - 0.1, self.max_position + 0.1)
            self.ax.set_ylim(0, 1.0)

        self.ax.clear()
        self.ax.set_xlim(self.min_position - 0.1, self.max_position + 0.1)
        self.ax.set_ylim(0, 1.0)

        # Draw mountain
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        self.ax.plot(xs, ys, 'b-', linewidth=2)

        # Draw car
        position, velocity = self.state
        height = self._height(position)
        self.ax.plot(position, height + 0.05, 'ro', markersize=10)

        # Draw goal
        self.ax.axvline(x=self.goal_position, color='g', linestyle='--', alpha=0.7)

        plt.pause(0.01)
        return self.fig

# Multi-Agent Environments
class PredatorPreyEnvironment(BaseEnvironment):
    """Multi-agent predator-prey environment"""

    def __init__(self, n_predators: int = 2, n_prey: int = 1,
                 grid_size: int = 10, max_steps: int = 100, seed: Optional[int] = None):
        super().__init__(seed)

        self.n_predators = n_predators
        self.n_prey = n_prey
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_space = spaces.Discrete(5)

        # Observation space: positions of all agents
        obs_dim = 2 * (n_predators + n_prey)
        self.observation_space = spaces.Box(low=0, high=grid_size-1,
                                          shape=(obs_dim,), dtype=np.int32)

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.step_count = 0

        # Initialize positions
        self.predator_positions = []
        self.prey_positions = []

        # Place predators
        for _ in range(self.n_predators):
            pos = self._get_random_position()
            self.predator_positions.append(pos)

        # Place prey
        for _ in range(self.n_prey):
            pos = self._get_random_position()
            while pos in self.predator_positions:
                pos = self._get_random_position()
            self.prey_positions.append(pos)

        return self._get_observation(), {}

    def _get_random_position(self) -> Tuple[int, int]:
        """Get random grid position"""
        return (self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size))

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        for pos in self.predator_positions + self.prey_positions:
            obs.extend(pos)
        return np.array(obs, dtype=np.int32)

    def step(self, actions: Dict[int, int]) -> Tuple[np.ndarray, Dict[int, float], bool, bool, Dict]:
        """Execute one time step"""
        self.step_count += 1

        # Move agents
        new_predator_positions = []
        for i, action in enumerate(actions['predators']):
            if i < len(self.predator_positions):
                new_pos = self._move_agent(self.predator_positions[i], action)
                new_predator_positions.append(new_pos)

        new_prey_positions = []
        for i, action in enumerate(actions['prey']):
            if i < len(self.prey_positions):
                new_pos = self._move_agent(self.prey_positions[i], action)
                new_prey_positions.append(new_pos)

        self.predator_positions = new_predator_positions
        self.prey_positions = new_prey_positions

        # Check captures
        rewards = {'predators': [0.0] * self.n_predators, 'prey': [0.0] * self.n_prey}
        captured_prey = []

        for i, predator_pos in enumerate(self.predator_positions):
            for j, prey_pos in enumerate(self.prey_positions):
                if predator_pos == prey_pos and j not in captured_prey:
                    rewards['predators'][i] += 10.0
                    rewards['prey'][j] -= 10.0
                    captured_prey.append(j)

        # Remove captured prey
        self.prey_positions = [pos for i, pos in enumerate(self.prey_positions)
                              if i not in captured_prey]

        # Step penalty
        for i in range(self.n_predators):
            rewards['predators'][i] -= 0.1
        for i in range(self.n_prey):
            rewards['prey'][i] -= 0.1

        # Check termination
        done = (self.step_count >= self.max_steps or
                len(self.prey_positions) == 0)

        return self._get_observation(), rewards, done, False, {}

    def _move_agent(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Move agent based on action"""
        x, y = position

        if action == 0:  # up
            y = min(y + 1, self.grid_size - 1)
        elif action == 1:  # down
            y = max(y - 1, 0)
        elif action == 2:  # left
            x = max(x - 1, 0)
        elif action == 3:  # right
            x = min(x + 1, self.grid_size - 1)
        # action == 4: stay

        return (x, y)

    def render(self, mode: str = 'human'):
        """Render environment"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True, alpha=0.3)

        # Draw predators (red triangles)
        for pos in self.predator_positions:
            self.ax.scatter(pos[0], pos[1], c='red', marker='^', s=200, alpha=0.8)

        # Draw prey (blue circles)
        for pos in self.prey_positions:
            self.ax.scatter(pos[0], pos[1], c='blue', marker='o', s=200, alpha=0.8)

        plt.pause(0.1)
        return self.fig

# Causal Environments
class CausalBanditEnvironment(BaseEnvironment):
    """Causal bandit environment for causal RL experiments"""

    def __init__(self, n_arms: int = 3, n_contexts: int = 2,
                 seed: Optional[int] = None):
        super().__init__(seed)

        self.n_arms = n_arms
        self.n_contexts = n_contexts

        # Generate causal structure
        self.causal_graph = self._generate_causal_graph()
        self.true_rewards = self._generate_true_rewards()

        self.action_space = spaces.Discrete(n_arms)
        self.observation_space = spaces.MultiDiscrete([n_contexts])

        self.reset()

    def _generate_causal_graph(self) -> nx.DiGraph:
        """Generate causal graph"""
        G = nx.DiGraph()

        # Add nodes
        G.add_node('context')
        for i in range(self.n_arms):
            G.add_node(f'arm_{i}')
        G.add_node('reward')

        # Add edges (context affects arms and reward)
        G.add_edge('context', 'reward')
        for i in range(self.n_arms):
            G.add_edge('context', f'arm_{i}')
            G.add_edge(f'arm_{i}', 'reward')

        return G

    def _generate_true_rewards(self) -> Dict[Tuple[int, int], float]:
        """Generate true reward function"""
        rewards = {}
        for context in range(self.n_contexts):
            for arm in range(self.n_arms):
                # Context-dependent rewards with some causal structure
                base_reward = np.sin(context + arm) * 0.5 + 0.5
                rewards[(context, arm)] = base_reward
        return rewards

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.context = self.np_random.integers(0, self.n_contexts)
        return np.array([self.context]), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        reward = self.true_rewards[(self.context, action)]
        reward += self.np_random.normal(0, 0.1)  # Add noise

        # Sample new context
        self.context = self.np_random.integers(0, self.n_contexts)

        done = False  # Bandits are episodic
        return np.array([self.context]), reward, done, False, {
            'true_reward': self.true_rewards[(self.context, action)]
        }

    def render(self, mode: str = 'human'):
        """Render causal graph"""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.ax.clear()
        pos = nx.spring_layout(self.causal_graph)
        nx.draw(self.causal_graph, pos, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=16, font_weight='bold', ax=self.ax)
        plt.title(f"Causal Bandit Environment (Context: {self.context})")
        plt.pause(0.01)
        return self.fig

# Quantum Environments
class QuantumControlEnvironment(BaseEnvironment):
    """Quantum control environment for quantum RL"""

    def __init__(self, n_qubits: int = 2, target_state: Optional[np.ndarray] = None,
                 max_steps: int = 100, seed: Optional[int] = None):
        super().__init__(seed)

        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self.max_steps = max_steps
        self.step_count = 0

        # Initialize target state (random if not provided)
        if target_state is None:
            self.target_state = self._random_state_vector()
        else:
            self.target_state = target_state / np.linalg.norm(target_state)

        # Action space: rotation angles for each qubit
        self.action_space = spaces.Box(low=-np.pi, high=np.pi,
                                     shape=(n_qubits * 3,), dtype=np.float32)  # RX, RY, RZ

        # Observation space: complex amplitudes (real and imaginary parts)
        self.observation_space = spaces.Box(low=-1, high=1,
                                          shape=(2 * self.state_dim,), dtype=np.float32)

        self.reset()

    def _random_state_vector(self) -> np.ndarray:
        """Generate random quantum state"""
        real = self.np_random.normal(0, 1, self.state_dim)
        imag = self.np_random.normal(0, 1, self.state_dim)
        state = real + 1j * imag
        return state / np.linalg.norm(state)

    def _get_pauli_matrices(self, qubit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Pauli matrices for a qubit"""
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        # Tensor product to get full matrices
        matrices = [I] * self.n_qubits
        matrices[qubit] = X
        RX = matrices[0]
        for mat in matrices[1:]:
            RX = np.kron(RX, mat)

        matrices = [I] * self.n_qubits
        matrices[qubit] = Y
        RY = matrices[0]
        for mat in matrices[1:]:
            RY = np.kron(RY, mat)

        matrices = [I] * self.n_qubits
        matrices[qubit] = Z
        RZ = matrices[0]
        for mat in matrices[1:]:
            RZ = np.kron(RZ, mat)

        return RX, RY, RZ

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.step_count = 0
        self.current_state = self._random_state_vector()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        return np.concatenate([self.current_state.real, self.current_state.imag])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        self.step_count += 1

        # Apply rotations to each qubit
        for qubit in range(self.n_qubits):
            rx_angle, ry_angle, rz_angle = action[qubit*3:(qubit+1)*3]
            RX, RY, RZ = self._get_pauli_matrices(qubit)

            # Apply rotations
            rotation = self._rotation_matrix(rx_angle, ry_angle, rz_angle, RX, RY, RZ)
            self.current_state = rotation @ self.current_state

        # Normalize state
        self.current_state = self.current_state / np.linalg.norm(self.current_state)

        # Compute fidelity reward
        fidelity = abs(np.vdot(self.target_state, self.current_state))**2
        reward = fidelity

        # Check termination
        done = self.step_count >= self.max_steps

        return self._get_observation(), reward, done, False, {
            'fidelity': fidelity,
            'target_reached': fidelity > 0.99
        }

    def _rotation_matrix(self, rx: float, ry: float, rz: float,
                        RX: np.ndarray, RY: np.ndarray, RZ: np.ndarray) -> np.ndarray:
        """Compute rotation matrix"""
        return np.cos(rx/2)*np.eye(self.state_dim) - 1j*np.sin(rx/2)*RX

    def render(self, mode: str = 'human'):
        """Render quantum state"""
        if not hasattr(self, 'fig'):
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot current state probabilities
        probs = np.abs(self.current_state)**2
        self.ax1.clear()
        self.ax1.bar(range(self.state_dim), probs, alpha=0.7, label='Current')
        self.ax1.set_xlabel('Basis State')
        self.ax1.set_ylabel('Probability')
        self.ax1.set_title('Current State')
        self.ax1.grid(True, alpha=0.3)

        # Plot target state probabilities
        target_probs = np.abs(self.target_state)**2
        self.ax2.clear()
        self.ax2.bar(range(self.state_dim), target_probs, alpha=0.7, color='orange', label='Target')
        self.ax2.set_xlabel('Basis State')
        self.ax2.set_ylabel('Probability')
        self.ax2.set_title('Target State')
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.01)
        return self.fig

# Federated Learning Environments
class FederatedLearningEnvironment(BaseEnvironment):
    """Federated learning environment"""

    def __init__(self, n_clients: int = 10, data_size: int = 1000,
                 heterogeneity: float = 0.5, seed: Optional[int] = None):
        super().__init__(seed)

        self.n_clients = n_clients
        self.data_size = data_size
        self.heterogeneity = heterogeneity

        # Generate heterogeneous data
        self.client_data = self._generate_client_data()

        # Action space: select clients to participate
        self.action_space = spaces.MultiBinary(n_clients)

        # Observation space: client statistics
        obs_dim = n_clients * 3  # mean, std, size for each client
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def _generate_client_data(self) -> List[Dict[str, np.ndarray]]:
        """Generate heterogeneous client data"""
        clients_data = []

        for i in range(self.n_clients):
            # Different data distributions for each client
            mean = self.np_random.normal(0, self.heterogeneity)
            std = 0.5 + self.np_random.random() * self.heterogeneity

            data = self.np_random.normal(mean, std, self.data_size)
            labels = (data > 0).astype(int)  # Binary classification

            clients_data.append({
                'features': data.reshape(-1, 1),
                'labels': labels,
                'mean': mean,
                'std': std,
                'size': self.data_size
            })

        return clients_data

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.global_model = np.zeros(1)  # Simple linear model
        self.round = 0
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        for client in self.client_data:
            obs.extend([client['mean'], client['std'], client['size']])
        return np.array(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one federated learning round"""
        selected_clients = np.where(action == 1)[0]

        if len(selected_clients) == 0:
            return self._get_observation(), -1.0, False, False, {'error': 'No clients selected'}

        # Aggregate updates from selected clients
        client_updates = []
        client_losses = []

        for client_idx in selected_clients:
            client = self.client_data[client_idx]

            # Simple local training (gradient descent step)
            features, labels = client['features'], client['labels']
            predictions = features @ self.global_model
            errors = predictions.flatten() - labels
            gradient = np.mean(errors * features.flatten())

            # Local update
            local_model = self.global_model - 0.01 * gradient
            client_updates.append(local_model)

            # Compute local loss
            local_loss = np.mean(errors**2)
            client_losses.append(local_loss)

        # Federated averaging
        self.global_model = np.mean(client_updates, axis=0)

        # Compute global loss
        global_loss = 0
        total_samples = 0

        for client in self.client_data:
            features, labels = client['features'], client['labels']
            predictions = features @ self.global_model
            errors = predictions.flatten() - labels
            global_loss += np.sum(errors**2)
            total_samples += len(labels)

        global_loss /= total_samples

        # Reward based on improvement and participation
        reward = -global_loss - 0.1 * len(selected_clients)  # Encourage efficiency

        self.round += 1
        done = self.round >= 100  # Max rounds

        return self._get_observation(), reward, done, False, {
            'global_loss': global_loss,
            'participation_rate': len(selected_clients) / self.n_clients,
            'selected_clients': selected_clients.tolist()
        }

    def render(self, mode: str = 'human'):
        """Render federated learning state"""
        if not hasattr(self, 'fig'):
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot client data distributions
        self.ax1.clear()
        for i, client in enumerate(self.client_data):
            data = client['features'].flatten()
            self.ax1.hist(data, alpha=0.3, bins=20, label=f'Client {i}')
        self.ax1.set_xlabel('Feature Value')
        self.ax1.set_ylabel('Frequency')
        self.ax1.set_title('Client Data Distributions')
        self.ax1.legend()

        # Plot global model
        self.ax2.clear()
        self.ax2.axvline(x=self.global_model[0], color='red', linewidth=3, label='Global Model')
        self.ax2.set_xlabel('Model Parameter')
        self.ax2.set_title(f'Global Model (Round {self.round})')
        self.ax2.legend()

        plt.tight_layout()
        plt.pause(0.01)
        return self.fig

print("✅ Environments module complete!")
print("Components implemented:")
print("- ContinuousMountainCar: Continuous control environment")
print("- PredatorPreyEnvironment: Multi-agent environment")
print("- CausalBanditEnvironment: Causal RL environment")
print("- QuantumControlEnvironment: Quantum RL environment")
print("- FederatedLearningEnvironment: Federated learning environment")
print("- BaseEnvironment: Base class for custom environments")