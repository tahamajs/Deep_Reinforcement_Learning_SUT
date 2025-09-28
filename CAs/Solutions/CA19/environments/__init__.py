"""
Environments Module for CA19 Advanced RL Systems

This module provides various environments for testing advanced RL algorithms,
including quantum-enhanced, neuromorphic, and hybrid systems.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Union
import random
import warnings

warnings.filterwarnings("ignore")


class NeuromorphicEnvironment(gym.Env):
    """
    Neuromorphic Environment for Event-Driven RL

    Environment designed for testing neuromorphic RL agents with event-driven
    sensory processing and real-time adaptation requirements.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 4,
        event_threshold: float = 0.1,
        sensor_noise: float = 0.05,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.event_threshold = event_threshold
        self.sensor_noise = sensor_noise

        # State space: continuous state variables
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim * 2,),  # State + event encoding
            dtype=np.float32,
        )

        # Action space: discrete actions
        self.action_space = spaces.Discrete(action_dim)

        # Environment dynamics
        self.state = None
        self.target_state = np.zeros(state_dim)
        self.velocity = np.zeros(state_dim)
        self.time_step = 0

        # Performance tracking
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.target_state = np.random.uniform(-0.5, 0.5, self.state_dim)
        self.velocity = np.zeros(self.state_dim)
        self.time_step = 0
        self.episode_reward = 0
        self.episode_length = 0

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step"""
        self.time_step += 1
        self.episode_length += 1

        # Convert action to force vector
        force = np.zeros(self.state_dim)
        if action < self.state_dim:
            force[action] = 1.0
        elif action < 2 * self.state_dim:
            force[action - self.state_dim] = -1.0

        # Update dynamics (simple physics)
        self.velocity += force * 0.1
        self.velocity *= 0.95  # Damping
        self.state += self.velocity

        # Add sensor noise
        noisy_state = self.state + np.random.normal(
            0, self.sensor_noise, self.state_dim
        )

        # Calculate reward (distance to target)
        distance = np.linalg.norm(noisy_state - self.target_state)
        reward = -distance

        # Check termination
        done = self.time_step >= 200 or distance < 0.1

        if done and distance < 0.1:
            reward += 10  # Bonus for reaching target

        self.episode_reward += reward

        info = {
            "distance_to_target": distance,
            "target_reached": distance < 0.1,
            "velocity_magnitude": np.linalg.norm(self.velocity),
            "episode_length": self.episode_length,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation with event encoding"""
        # State variables
        state_obs = self.state.copy()

        # Event encoding (detect significant changes)
        events = np.abs(self.velocity) > self.event_threshold
        event_obs = events.astype(float)

        # Combine state and events
        observation = np.concatenate([state_obs, event_obs])

        return observation

    def render(self, mode: str = "human"):
        """Render environment state"""
        if mode == "human":
            print(f"State: {self.state}")
            print(f"Target: {self.target_state}")
            print(f"Distance: {np.linalg.norm(self.state - self.target_state):.3f}")
            print(f"Velocity: {self.velocity}")
            print(f"Time: {self.time_step}")


class HybridQuantumClassicalEnvironment(gym.Env):
    """
    Environment for Testing Hybrid Quantum-Classical RL Systems

    Features complex state spaces that benefit from quantum-enhanced
    representation while requiring classical temporal processing.
    """

    def __init__(
        self, state_dim: int = 8, action_dim: int = 16, quantum_complexity: float = 0.7
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quantum_complexity = quantum_complexity

        # Complex observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(action_dim)

        # Environment state
        self.state = None
        self.hidden_state = None
        self.targets = []
        self.obstacles = []

        # Dynamics parameters
        self.dt = 0.1
        self.friction = 0.98
        self.max_force = 2.0

    def reset(self) -> np.ndarray:
        """Reset to initial state"""
        self.state = np.random.uniform(-2, 2, self.state_dim // 2)
        self.hidden_state = np.random.uniform(-1, 1, self.state_dim // 2)

        # Generate targets and obstacles
        self.targets = [
            np.random.uniform(-1.5, 1.5, self.state_dim // 2)
            for _ in range(np.random.randint(2, 5))
        ]
        self.obstacles = [
            np.random.uniform(-1.5, 1.5, self.state_dim // 2)
            for _ in range(np.random.randint(1, 4))
        ]

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in environment"""
        # Decode action
        force = self._decode_action(action)

        # Update visible state
        self.state += force * self.dt
        self.state *= self.friction

        # Update hidden state (more complex dynamics)
        self.hidden_state += np.sin(self.state) * self.dt * self.quantum_complexity
        self.hidden_state += np.random.normal(0, 0.1, len(self.hidden_state))

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        done = self._check_termination()

        info = {
            "targets_reached": self._count_targets_reached(),
            "obstacles_hit": self._count_obstacles_hit(),
            "state_complexity": self._calculate_state_complexity(),
            "quantum_advantage_potential": self.quantum_complexity,
        }

        return self._get_observation(), reward, done, info

    def _decode_action(self, action: int) -> np.ndarray:
        """Decode discrete action to continuous force vector"""
        force = np.zeros(self.state_dim // 2)

        # Binary encoding of force directions
        for i in range(len(force)):
            if action & (1 << i):
                force[i] = self.max_force
            elif action & (1 << (i + len(force))):
                force[i] = -self.max_force

        return force

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Combine visible and hidden state
        observation = np.concatenate([self.state, self.hidden_state])

        # Add quantum-inspired features (correlations, phases)
        if self.quantum_complexity > 0.5:
            # Add phase information
            phases = np.angle(self.state + 1j * self.hidden_state)
            observation = np.concatenate([observation, phases])

        return observation

    def _calculate_reward(self) -> float:
        """Calculate reward based on targets and obstacles"""
        reward = 0

        # Target rewards
        for target in self.targets:
            distance = np.linalg.norm(self.state - target)
            if distance < 0.3:
                reward += 1.0 / (distance + 0.1)
            else:
                reward -= distance * 0.1

        # Obstacle penalties
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.state - obstacle)
            if distance < 0.2:
                reward -= 5.0

        # Complexity bonus (reward for handling complex dynamics)
        complexity_bonus = self.quantum_complexity * np.var(self.hidden_state)
        reward += complexity_bonus * 0.1

        return reward

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Time limit
        if not hasattr(self, "step_count"):
            self.step_count = 0
        self.step_count += 1

        if self.step_count >= 500:
            return True

        # Success condition (reach all targets)
        targets_reached = self._count_targets_reached()
        if targets_reached >= len(self.targets):
            return True

        # Failure condition (hit too many obstacles)
        obstacles_hit = self._count_obstacles_hit()
        if obstacles_hit >= len(self.obstacles):
            return True

        return False

    def _count_targets_reached(self) -> int:
        """Count how many targets have been reached"""
        reached = 0
        for target in self.targets:
            if np.linalg.norm(self.state - target) < 0.3:
                reached += 1
        return reached

    def _count_obstacles_hit(self) -> int:
        """Count how many obstacles have been hit"""
        hit = 0
        for obstacle in self.obstacles:
            if np.linalg.norm(self.state - obstacle) < 0.2:
                hit += 1
        return hit

    def _calculate_state_complexity(self) -> float:
        """Calculate complexity measure of current state"""
        # Measure correlations and non-linearities
        visible_complexity = np.var(self.state) * np.mean(np.abs(self.state))
        hidden_complexity = np.var(self.hidden_state) * np.mean(
            np.abs(self.hidden_state)
        )

        # Cross-correlations
        correlation = (
            np.corrcoef(self.state, self.hidden_state)[0, 1]
            if len(self.state) > 1
            else 0
        )

        complexity = (visible_complexity + hidden_complexity) * (1 + abs(correlation))
        return complexity


class MetaLearningEnvironment(gym.Env):
    """
    Environment for Testing Meta-Learning Capabilities

    Features changing task distributions and requires agents to adapt
    quickly to new scenarios.
    """

    def __init__(
        self, base_state_dim: int = 6, num_tasks: int = 5, task_change_prob: float = 0.1
    ):
        super().__init__()

        self.base_state_dim = base_state_dim
        self.num_tasks = num_tasks
        self.task_change_prob = task_change_prob

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_state_dim + 2,),  # State + task context
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(8)

        # Task parameters
        self.current_task = 0
        self.task_params = self._generate_task_parameters()
        self.state = None
        self.episode_count = 0

    def _generate_task_parameters(self) -> List[Dict]:
        """Generate parameters for different tasks"""
        tasks = []
        for i in range(self.num_tasks):
            task = {
                "dynamics_matrix": np.random.uniform(
                    -1, 1, (self.base_state_dim, self.base_state_dim)
                ),
                "control_matrix": np.random.uniform(-2, 2, (self.base_state_dim, 3)),
                "target_function": lambda x: np.sin(x * (i + 1) * np.pi / 4),
                "reward_scale": np.random.uniform(0.5, 2.0),
                "noise_level": np.random.uniform(0.01, 0.1),
            }
            tasks.append(task)
        return tasks

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.episode_count += 1

        # Possibly change task
        if np.random.random() < self.task_change_prob:
            self.current_task = np.random.randint(self.num_tasks)

        # Initialize state
        self.state = np.random.uniform(-1, 1, self.base_state_dim)

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step in current task"""
        task = self.task_params[self.current_task]

        # Decode action
        action_vector = np.zeros(3)
        action_vector[action % 3] = 1 if action < 4 else -1

        # Apply task dynamics
        control_effect = task["control_matrix"] @ action_vector
        dynamics_effect = task["dynamics_matrix"] @ self.state

        # Update state
        self.state += (dynamics_effect + control_effect) * 0.1
        self.state += np.random.normal(0, task["noise_level"], self.base_state_dim)

        # Clip state
        self.state = np.clip(self.state, -3, 3)

        # Calculate reward
        target_value = task["target_function"](np.linalg.norm(self.state))
        current_value = np.mean(self.state)
        reward = -abs(current_value - target_value) * task["reward_scale"]

        # Check termination
        done = (
            abs(current_value - target_value) < 0.1 or np.linalg.norm(self.state) > 2.5
        )

        info = {
            "current_task": self.current_task,
            "task_changed": False,  # Would be set in reset
            "target_value": target_value,
            "current_value": current_value,
            "adaptation_required": self.task_change_prob > 0,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get observation with task context"""
        task_context = np.array(
            [
                self.current_task / self.num_tasks,  # Normalized task ID
                self.episode_count / 100,  # Episode progress
            ]
        )

        return np.concatenate([self.state, task_context])


class ContinualLearningEnvironment(gym.Env):
    """
    Environment for Testing Continual Learning Capabilities

    Features sequential task learning with catastrophic forgetting challenges.
    """

    def __init__(
        self, state_dim: int = 4, num_phases: int = 3, phase_length: int = 100
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_phases = num_phases
        self.phase_length = phase_length

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        # Phase-specific parameters
        self.current_phase = 0
        self.phase_step = 0
        self.phase_params = self._generate_phase_parameters()

        self.state = None

    def _generate_phase_parameters(self) -> List[Dict]:
        """Generate different parameters for each learning phase"""
        phases = []
        for i in range(self.num_phases):
            phase = {
                "state_dynamics": np.random.uniform(
                    -1, 1, (self.state_dim, self.state_dim)
                ),
                "action_effects": np.random.uniform(-2, 2, (self.state_dim, 4)),
                "reward_function": self._create_reward_function(i),
                "optimal_policy": np.random.randint(
                    0, 4, 10
                ),  # Sequence of optimal actions
            }
            phases.append(phase)
        return phases

    def _create_reward_function(self, phase_idx: int):
        """Create phase-specific reward function"""
        if phase_idx == 0:
            return lambda s, a: -np.linalg.norm(s) + (a == 0) * 2
        elif phase_idx == 1:
            return lambda s, a: np.sum(s) + (a == 1) * 2
        else:
            return lambda s, a: -np.var(s) + (a == 2) * 2

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.phase_step += 1

        # Check for phase transition
        if self.phase_step >= self.phase_length:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.phase_step = 0

        # Initialize state for current phase
        self.state = np.random.uniform(-1, 1, self.state_dim)

        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step in current phase"""
        phase = self.phase_params[self.current_phase]

        # Apply action effects
        action_effect = phase["action_effects"][:, action]
        dynamics_effect = phase["state_dynamics"] @ self.state

        # Update state
        self.state += (dynamics_effect + action_effect) * 0.1
        self.state += np.random.normal(0, 0.05, self.state_dim)
        self.state = np.clip(self.state, -2, 2)

        # Calculate reward
        reward = phase["reward_function"](self.state, action)

        # Check termination
        done = self.phase_step >= self.phase_length - 1

        info = {
            "current_phase": self.current_phase,
            "phase_progress": self.phase_step / self.phase_length,
            "phase_changed": self.phase_step == 0 and self.current_phase > 0,
            "state_norm": np.linalg.norm(self.state),
        }

        return self.state, reward, done, info


class HierarchicalEnvironment(gym.Env):
    """
    Environment for Testing Hierarchical RL Capabilities

    Features multiple levels of abstraction with temporal dependencies.
    """

    def __init__(self, state_dim: int = 6, num_levels: int = 3):
        super().__init__()

        self.state_dim = state_dim
        self.num_levels = num_levels

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim + num_levels,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(8)

        # Hierarchical state
        self.level_states = [
            np.zeros(state_dim // num_levels) for _ in range(num_levels)
        ]
        self.level_goals = [
            np.random.uniform(-1, 1, state_dim // num_levels) for _ in range(num_levels)
        ]
        self.current_level = 0

        self.state = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        """Reset hierarchical environment"""
        for i in range(self.num_levels):
            self.level_states[i] = np.random.uniform(-1, 1, len(self.level_states[i]))
            self.level_goals[i] = np.random.uniform(-1, 1, len(self.level_states[i]))

        self.current_level = 0
        self.step_count = 0

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute hierarchical step"""
        self.step_count += 1

        # Decode action for current level
        level_action = action % 4
        level_command = action // 4  # 0: stay, 1: ascend, 2: descend

        # Update current level state
        current_state = self.level_states[self.current_level]
        goal = self.level_goals[self.current_level]

        # Apply action to current level
        action_vector = np.zeros(len(current_state))
        if level_action < len(current_state):
            action_vector[level_action] = 1.0

        current_state += action_vector * 0.2
        current_state += (goal - current_state) * 0.1  # Goal attraction

        # Level transitions
        if level_command == 1 and self.current_level < self.num_levels - 1:
            self.current_level += 1
        elif level_command == 2 and self.current_level > 0:
            self.current_level -= 1

        # Calculate hierarchical reward
        reward = self._calculate_hierarchical_reward()

        # Check termination
        done = self.step_count >= 300 or self._check_hierarchy_complete()

        info = {
            "current_level": self.current_level,
            "level_progress": [
                np.linalg.norm(s - g)
                for s, g in zip(self.level_states, self.level_goals)
            ],
            "hierarchy_complete": self._check_hierarchy_complete(),
            "level_transitions": level_command > 0,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get hierarchical observation"""
        # Concatenate all level states
        state_obs = np.concatenate(self.level_states)

        # Add level indicators
        level_indicators = np.zeros(self.num_levels)
        level_indicators[self.current_level] = 1.0

        return np.concatenate([state_obs, level_indicators])

    def _calculate_hierarchical_reward(self) -> float:
        """Calculate reward based on hierarchical progress"""
        reward = 0

        # Local level reward
        current_state = self.level_states[self.current_level]
        current_goal = self.level_goals[self.current_level]
        local_distance = np.linalg.norm(current_state - current_goal)
        reward -= local_distance

        # Hierarchical bonus
        if local_distance < 0.3:
            reward += 1.0

        # Global hierarchy reward
        total_progress = sum(
            np.linalg.norm(s - g) for s, g in zip(self.level_states, self.level_goals)
        )
        reward -= total_progress * 0.1

        return reward

    def _check_hierarchy_complete(self) -> bool:
        """Check if all levels are complete"""
        return all(
            np.linalg.norm(s - g) < 0.3
            for s, g in zip(self.level_states, self.level_goals)
        )
