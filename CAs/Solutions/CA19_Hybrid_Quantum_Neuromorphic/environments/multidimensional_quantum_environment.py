"""
Multidimensional Quantum Environment

This module implements highly complex environments for testing advanced RL algorithms:
- Multi-dimensional state spaces with quantum entanglement
- Dynamic reward landscapes with quantum interference
- Temporal dynamics with decoherence effects
- Hierarchical task structures
- Adaptive difficulty scaling
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
import random
import warnings
from scipy.linalg import expm
from scipy.special import expit

warnings.filterwarnings("ignore")


class QuantumStateSpace:
    """
    Complex quantum state space with entanglement and interference
    """

    def __init__(self, n_dimensions: int = 16, n_qubits: int = 8):
        self.n_dimensions = n_dimensions
        self.n_qubits = n_qubits
        self.dimension = 2**n_qubits

        # Quantum state representation
        self.state = np.ones(self.dimension, dtype=complex) / np.sqrt(self.dimension)

        # Entanglement matrices
        self.entanglement_matrices = []
        self._initialize_entanglement_matrices()

        # Interference patterns
        self.interference_patterns = np.random.randn(n_dimensions, n_dimensions)

        # Decoherence parameters
        self.decoherence_rate = 0.01
        self.coherence_time = 1.0

    def _initialize_entanglement_matrices(self):
        """Initialize random entanglement matrices"""
        for _ in range(self.n_qubits // 2):
            # Create random unitary matrix for entanglement
            random_matrix = np.random.randn(self.dimension, self.dimension)
            unitary_matrix = expm(1j * (random_matrix + random_matrix.T))
            self.entanglement_matrices.append(unitary_matrix)

    def apply_quantum_evolution(self, time_step: float):
        """Apply quantum evolution with decoherence"""
        # Apply entanglement
        for matrix in self.entanglement_matrices:
            self.state = matrix @ self.state

        # Apply decoherence
        decoherence_factor = np.exp(-time_step / self.coherence_time)
        self.state *= decoherence_factor

        # Normalize
        norm = np.sqrt(np.sum(np.abs(self.state) ** 2))
        if norm > 0:
            self.state = self.state / norm

    def measure_state(self) -> np.ndarray:
        """Measure quantum state and return classical observables"""
        probabilities = np.abs(self.state) ** 2

        # Sample measurement outcomes
        measurement_indices = np.random.choice(
            len(probabilities), size=self.n_dimensions, p=probabilities, replace=True
        )

        # Convert to classical state
        classical_state = np.zeros(self.n_dimensions)
        for i, idx in enumerate(measurement_indices):
            classical_state[i] = (idx / self.dimension) * 2 - 1  # Normalize to [-1, 1]

        return classical_state


class DynamicRewardLandscape:
    """
    Complex reward landscape with quantum interference patterns
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Multiple reward components
        self.reward_components = []
        self._initialize_reward_components()

        # Temporal dynamics
        self.time_phase = 0.0
        self.phase_velocity = 0.1

        # Interference patterns
        self.interference_frequency = 2.0
        self.interference_amplitude = 0.5

    def _initialize_reward_components(self):
        """Initialize different reward components"""
        # Sparse reward component
        sparse_positions = np.random.choice(self.state_dim, size=5, replace=False)
        sparse_weights = np.random.randn(5)
        self.reward_components.append(("sparse", sparse_positions, sparse_weights))

        # Dense reward component
        dense_weights = np.random.randn(self.state_dim) * 0.1
        self.reward_components.append(("dense", None, dense_weights))

        # Quadratic reward component
        quadratic_matrix = np.random.randn(self.state_dim, self.state_dim)
        quadratic_matrix = (quadratic_matrix + quadratic_matrix.T) / 2
        self.reward_components.append(("quadratic", quadratic_matrix, None))

        # Periodic reward component
        periodic_frequencies = np.random.randn(self.state_dim)
        periodic_phases = np.random.randn(self.state_dim) * 2 * np.pi
        self.reward_components.append(
            ("periodic", periodic_frequencies, periodic_phases)
        )

    def calculate_reward(
        self, state: np.ndarray, action: int, time_step: float
    ) -> float:
        """Calculate complex reward with interference"""
        self.time_phase += self.phase_velocity

        total_reward = 0.0

        # Sparse component
        sparse_type, positions, weights = self.reward_components[0]
        sparse_reward = np.sum(weights * state[positions])
        total_reward += 0.3 * sparse_reward

        # Dense component
        dense_type, _, weights = self.reward_components[1]
        dense_reward = np.dot(weights, state)
        total_reward += 0.2 * dense_reward

        # Quadratic component
        quad_type, matrix, _ = self.reward_components[2]
        quad_reward = np.dot(state, np.dot(matrix, state))
        total_reward += 0.2 * quad_reward

        # Periodic component
        periodic_type, frequencies, phases = self.reward_components[3]
        periodic_reward = np.sum(np.sin(frequencies * self.time_phase + phases) * state)
        total_reward += 0.1 * periodic_reward

        # Action-dependent reward
        action_reward = np.sin(action * 0.5 + self.time_phase) * 0.2
        total_reward += action_reward

        # Interference pattern
        interference = self.interference_amplitude * np.sin(
            self.interference_frequency * self.time_phase
            + np.dot(state, np.ones(self.state_dim)) * 0.5
        )
        total_reward += interference

        return total_reward


class HierarchicalTaskStructure:
    """
    Hierarchical task structure with multiple levels of complexity
    """

    def __init__(self, n_levels: int = 4, tasks_per_level: int = 3):
        self.n_levels = n_levels
        self.tasks_per_level = tasks_per_level
        self.total_tasks = n_levels * tasks_per_level

        # Task hierarchy
        self.current_level = 0
        self.current_task = 0
        self.task_completion = np.zeros(self.total_tasks)

        # Task complexity scaling
        self.complexity_factors = np.linspace(0.5, 2.0, self.total_tasks)

        # Inter-task dependencies
        self.task_dependencies = self._create_task_dependencies()

        # Dynamic task switching
        self.task_switch_probability = 0.1
        self.task_switch_timer = 0

    def _create_task_dependencies(self) -> np.ndarray:
        """Create dependency matrix between tasks"""
        dependencies = np.zeros((self.total_tasks, self.total_tasks))

        for i in range(self.total_tasks):
            level_i = i // self.tasks_per_level
            task_i = i % self.tasks_per_level

            # Dependencies within same level
            for j in range(self.total_tasks):
                if i != j:
                    level_j = j // self.tasks_per_level
                    task_j = j % self.tasks_per_level

                    if level_i == level_j and abs(task_i - task_j) == 1:
                        dependencies[i, j] = 0.3
                    elif level_i == level_j + 1 and task_i == task_j:
                        dependencies[i, j] = 0.7

        return dependencies

    def update_task_state(self, reward: float, time_step: float):
        """Update task completion and potentially switch tasks"""
        # Update completion for current task
        current_task_idx = self.current_level * self.tasks_per_level + self.current_task
        self.task_completion[current_task_idx] += reward * 0.01

        # Check for task completion
        if self.task_completion[current_task_idx] >= 1.0:
            self._complete_task(current_task_idx)

        # Dynamic task switching
        self.task_switch_timer += time_step
        if (
            self.task_switch_timer > 10.0
            and np.random.random() < self.task_switch_probability
        ):
            self._switch_task()
            self.task_switch_timer = 0

    def _complete_task(self, task_idx: int):
        """Handle task completion"""
        self.task_completion[task_idx] = 1.0

        # Boost dependent tasks
        for i, dependency in enumerate(self.task_dependencies[task_idx]):
            if dependency > 0:
                self.task_completion[i] += dependency * 0.1
                self.task_completion[i] = np.clip(self.task_completion[i], 0, 1)

    def _switch_task(self):
        """Switch to a new task"""
        # Choose new task based on completion status
        incomplete_tasks = np.where(self.task_completion < 0.5)[0]
        if len(incomplete_tasks) > 0:
            new_task_idx = np.random.choice(incomplete_tasks)
            self.current_level = new_task_idx // self.tasks_per_level
            self.current_task = new_task_idx % self.tasks_per_level

    def get_task_complexity(self) -> float:
        """Get current task complexity factor"""
        current_task_idx = self.current_level * self.tasks_per_level + self.current_task
        base_complexity = self.complexity_factors[current_task_idx]

        # Adjust based on task completion
        completion_factor = 1.0 + (1.0 - self.task_completion[current_task_idx]) * 0.5

        return base_complexity * completion_factor


class MultidimensionalQuantumEnvironment(gym.Env):
    """
    Highly complex multidimensional quantum environment
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 16,
        n_qubits: int = 8,
        n_task_levels: int = 4,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)

        # Core components
        self.quantum_state_space = QuantumStateSpace(state_dim, n_qubits)
        self.reward_landscape = DynamicRewardLandscape(state_dim, action_dim)
        self.task_structure = HierarchicalTaskStructure(n_task_levels)

        # Environment state
        self.state = None
        self.time_step = 0.0
        self.episode_length = 0
        self.max_episode_length = 1000

        # Performance tracking
        self.episode_rewards = []
        self.quantum_metrics = []
        self.task_performance = []

        # Adaptive difficulty
        self.difficulty_level = 1.0
        self.difficulty_adaptation_rate = 0.001

        # Crisis events
        self.crisis_probability = 0.05
        self.crisis_strength = 0.5
        self.crisis_active = False
        self.crisis_timer = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset quantum state
        self.quantum_state_space.state = np.ones(
            self.quantum_state_space.dimension, dtype=complex
        ) / np.sqrt(self.quantum_state_space.dimension)

        # Reset time and episode state
        self.time_step = 0.0
        self.episode_length = 0
        self.crisis_active = False
        self.crisis_timer = 0

        # Get initial state from quantum measurement
        self.state = self.quantum_state_space.measure_state()

        # Reset task structure
        self.task_structure.current_level = 0
        self.task_structure.current_task = 0

        # Initial info
        info = {
            "quantum_fidelity": 1.0,
            "entanglement_measure": 0.0,
            "task_level": self.task_structure.current_level,
            "task_completion": self.task_structure.task_completion.copy(),
            "difficulty_level": self.difficulty_level,
            "crisis_active": False,
        }

        return self.state.astype(np.float32), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        self.episode_length += 1
        self.time_step += 0.01

        # Apply quantum evolution
        self.quantum_state_space.apply_quantum_evolution(self.time_step)

        # Calculate reward
        reward = self.reward_landscape.calculate_reward(
            self.state, action, self.time_step
        )

        # Apply task structure effects
        task_complexity = self.task_structure.get_task_complexity()
        reward *= task_complexity

        # Check for crisis events
        crisis_reward_modifier = self._handle_crisis_events()
        reward *= crisis_reward_modifier

        # Apply difficulty scaling
        reward *= self.difficulty_level

        # Update task structure
        self.task_structure.update_task_state(reward, self.time_step)

        # Get new state
        self.state = self.quantum_state_space.measure_state()

        # Calculate quantum metrics
        quantum_fidelity = np.mean(np.abs(self.quantum_state_space.state))
        entanglement_measure = self._calculate_entanglement()

        # Check termination conditions
        terminated = self.episode_length >= self.max_episode_length
        truncated = False

        # Adaptive difficulty adjustment
        self._adjust_difficulty(reward)

        # Store metrics
        self.episode_rewards.append(reward)
        self.quantum_metrics.append(
            {"fidelity": quantum_fidelity, "entanglement": entanglement_measure}
        )
        self.task_performance.append(
            {
                "level": self.task_structure.current_level,
                "task": self.task_structure.current_task,
                "completion": self.task_structure.task_completion.copy(),
            }
        )

        # Create info dictionary
        info = {
            "quantum_fidelity": quantum_fidelity,
            "entanglement_measure": entanglement_measure,
            "task_level": self.task_structure.current_level,
            "task_completion": self.task_structure.task_completion.copy(),
            "difficulty_level": self.difficulty_level,
            "crisis_active": self.crisis_active,
            "episode_length": self.episode_length,
            "total_reward": sum(self.episode_rewards),
            "avg_reward": (
                np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            ),
        }

        return self.state.astype(np.float32), reward, terminated, truncated, info

    def _handle_crisis_events(self) -> float:
        """Handle crisis events that affect the environment"""
        crisis_modifier = 1.0

        # Check for new crisis
        if not self.crisis_active and np.random.random() < self.crisis_probability:
            self.crisis_active = True
            self.crisis_timer = np.random.randint(10, 50)

        # Handle active crisis
        if self.crisis_active:
            self.crisis_timer -= 1
            crisis_modifier = 1.0 - self.crisis_strength

            # End crisis
            if self.crisis_timer <= 0:
                self.crisis_active = False
                crisis_modifier = 1.0 + self.crisis_strength  # Bonus after crisis

        return crisis_modifier

    def _calculate_entanglement(self) -> float:
        """Calculate entanglement measure"""
        if self.quantum_state_space.n_qubits < 2:
            return 0.0

        # Simplified entanglement calculation
        state = self.quantum_state_space.state
        probabilities = np.abs(state) ** 2

        # Shannon entropy as entanglement measure
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(probabilities))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _adjust_difficulty(self, reward: float):
        """Adaptively adjust environment difficulty"""
        # Increase difficulty if performance is too good
        if len(self.episode_rewards) > 100:
            recent_performance = np.mean(self.episode_rewards[-100:])
            if recent_performance > 1.0:
                self.difficulty_level += self.difficulty_adaptation_rate
            elif recent_performance < -1.0:
                self.difficulty_level -= self.difficulty_adaptation_rate

        # Clamp difficulty level
        self.difficulty_level = np.clip(self.difficulty_level, 0.1, 3.0)

    def get_environment_metrics(self) -> Dict[str, Union[float, List[float]]]:
        """Get comprehensive environment metrics"""
        if not self.episode_rewards:
            return {}

        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "reward_std": np.std(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "current_difficulty": self.difficulty_level,
            "task_completion_rate": np.mean(self.task_structure.task_completion),
            "quantum_fidelity_avg": np.mean(
                [m["fidelity"] for m in self.quantum_metrics]
            ),
            "entanglement_avg": np.mean(
                [m["entanglement"] for m in self.quantum_metrics]
            ),
            "crisis_frequency": (
                sum(1 for perf in self.task_performance if "crisis" in perf)
                / len(self.task_performance)
                if self.task_performance
                else 0.0
            ),
        }

    def render(self, mode: str = "human"):
        """Render environment state"""
        if mode == "human":
            print(f"Episode Length: {self.episode_length}")
            print(
                f"Current Reward: {self.episode_rewards[-1] if self.episode_rewards else 0.0}"
            )
            print(f"Difficulty Level: {self.difficulty_level:.3f}")
            print(f"Task Level: {self.task_structure.current_level}")
            print(
                f"Quantum Fidelity: {np.mean(np.abs(self.quantum_state_space.state)):.3f}"
            )
            print(f"Crisis Active: {self.crisis_active}")

    def close(self):
        """Close environment"""
        pass
