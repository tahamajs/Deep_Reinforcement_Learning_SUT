"""
Advanced Complex Environments
محیط‌های پیچیده و پیشرفته

This module contains complex environments including:
- Dynamic Multi-Objective Environments
- Partially Observable Environments
- Continuous Control Environments
- Realistic Physics Simulations
- Adversarial Environments
- Multi-Modal Environments
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import random
import math
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """
    Configuration for complex environments.

    Attributes:
        size (int): Size of the grid-world environment (e.g., size x size). Defaults to 10.
        num_agents (int): Number of agents in the environment. Defaults to 3.
        num_objectives (int): Number of objectives in multi-objective environments. Defaults to 2.
        observation_noise (float): Standard deviation of noise added to observations. Defaults to 0.1.
        action_noise (float): Standard deviation of noise added to actions. Defaults to 0.05.
        dynamic_changes (bool): Whether the environment undergoes dynamic changes over time. Defaults to True.
        partial_observability (bool): Whether agents have partial observability. Defaults to True.
        physics_enabled (bool): Whether basic physics (e.g., momentum) are enabled. Defaults to True.
        adversarial_mode (bool): Whether an adversarial agent is active in the environment. Defaults to False.
    """

    size: int = 10
    num_agents: int = 3
    num_objectives: int = 2
    observation_noise: float = 0.1
    action_noise: float = 0.05
    dynamic_changes: bool = True
    partial_observability: bool = True
    physics_enabled: bool = True
    adversarial_mode: bool = False


class DynamicMultiObjectiveEnvironment:
    """
    Dynamic Multi-Objective Environment with changing goals.

    This environment features multiple agents, dynamic objectives, obstacles, and hazards.
    It simulates a complex, non-stationary setting suitable for advanced RL research.

    Args:
        config (EnvironmentConfig): Configuration object for the environment settings.
    """

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.grid_size = config.size
        self.num_agents = config.num_agents
        self.num_objectives = config.num_objectives
        self.dynamic_changes = config.dynamic_changes
        self.physics_enabled = config.physics_enabled

        self.agent_positions: List[List[int]] = []
        self.agent_velocities: List[List[float]] = []
        self.objectives: List[Dict[str, Any]] = []
        self.obstacles: List[List[int]] = []
        self.hazards: List[List[int]] = []
        self.current_step = 0
        self.max_steps = self.grid_size * 20

        self.reset()

    def _generate_obstacles(self) -> List[List[int]]:
        """
        Generate random obstacle positions within the grid.

        Returns:
            List[List[int]]: A list of [x, y] coordinates for obstacles.
        """
        num_obstacles = self.grid_size // 3
        obstacles = []
        for _ in range(num_obstacles):
            obstacles.append([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        return obstacles

    def _generate_resources(self) -> List[Dict[str, Any]]:
        """
        Generate random resource positions and values.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each with 'position' ([x, y]) and 'value' (float).
        """
        num_resources = self.grid_size // 2
        resources = []
        for _ in range(num_resources):
            pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            resources.append({"position": pos, "value": random.uniform(0.1, 1.0)})
        return resources

    def _generate_hazards(self) -> List[List[int]]:
        """
        Generate random hazard zone positions.

        Returns:
            List[List[int]]: A list of [x, y] coordinates for hazards.
        """
        num_hazards = self.grid_size // 4
        hazards = []
        for _ in range(num_hazards):
            hazards.append([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        return hazards

    def _update_objectives(self):
        """
        Dynamically update objective positions or values.
        Called if `self.dynamic_changes` is True.
        """
        if self.dynamic_changes and self.current_step % 10 == 0:
            for obj in self.objectives:
                obj["position"] = [
                    (obj["position"][0] + random.choice([-1, 0, 1])) % self.grid_size,
                    (obj["position"][1] + random.choice([-1, 0, 1])) % self.grid_size,
                ]
                obj["value"] = max(0.1, obj["value"] + random.uniform(-0.1, 0.1))

    def _apply_physics(self, agent_id: int, action: np.ndarray) -> np.ndarray:
        """
        Apply simplified physics to update agent velocity and position.

        Args:
            agent_id (int): The ID of the agent.
            action (np.ndarray): The action taken by the agent (e.g., [dx, dy] force).

        Returns:
            np.ndarray: The new position of the agent [x, y].
        """
        if not self.physics_enabled:
            move = np.array([0, 0])
            if action == 0:  # Up
                move[1] = -1
            elif action == 1:  # Down
                move[1] = 1
            elif action == 2:  # Left
                move[0] = -1
            elif action == 3:  # Right
                move[0] = 1
            # Apply action noise
            move = (move + np.random.normal(0, self.config.action_noise, size=2)).astype(int)

            new_pos = [
                (self.agent_positions[agent_id][0] + move[0]) % self.grid_size,
                (self.agent_positions[agent_id][1] + move[1]) % self.grid_size,
            ]
            return np.array(new_pos)

        # Simple physics: action influences velocity, velocity influences position
        acceleration = action[:2] * 0.1  # Assume action is a 2D force vector for continuous control
        self.agent_velocities[agent_id] = [
            self.agent_velocities[agent_id][0] + acceleration[0],
            self.agent_velocities[agent_id][1] + acceleration[1],
        ]
        # Add damping
        self.agent_velocities[agent_id] = [
            v * 0.9 for v in self.agent_velocities[agent_id]
        ]

        new_pos = [
            (self.agent_positions[agent_id][0] + int(self.agent_velocities[agent_id][0])) % self.grid_size,
            (self.agent_positions[agent_id][1] + int(self.agent_velocities[agent_id][1])) % self.grid_size,
        ]
        return np.array(new_pos)

    def _check_collisions(self) -> List[bool]:
        """
        Check for collisions between agents and obstacles.

        Returns:
            List[bool]: A list of booleans, True if agent i collided, False otherwise.
        """
        collisions = [False] * self.num_agents
        for i, pos in enumerate(self.agent_positions):
            if pos in self.obstacles:
                collisions[i] = True
            for j, other_pos in enumerate(self.agent_positions):
                if i != j and pos == other_pos:
                    collisions[i] = True  # Agent-agent collision
        return collisions

    def _check_hazards(self) -> List[bool]:
        """
        Check if agents are in hazard zones.

        Returns:
            List[bool]: A list of booleans, True if agent i is in a hazard zone, False otherwise.
        """
        in_hazard = [False] * self.num_agents
        for i, pos in enumerate(self.agent_positions):
            if pos in self.hazards:
                in_hazard[i] = True
        return in_hazard

    def _collect_resources(self) -> List[float]:
        """
        Check if agents collect any resources and return rewards.

        Returns:
            List[float]: A list of rewards, one for each agent.
        """
        rewards = [0.0] * self.num_agents
        for i, pos in enumerate(self.agent_positions):
            for resource in self.resources:
                if pos == resource["position"]:
                    rewards[i] += resource["value"]
                    self.resources.remove(resource) # Remove collected resource
                    self.resources.extend(self._generate_resources()) # Respawn
        return rewards

    def reset(self) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple[List[np.ndarray], Dict[str, Any]]: A tuple containing initial observations for all agents
                                                   and an info dictionary.
        """
        self.agent_positions = [
            [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            for _ in range(self.num_agents)
        ]
        self.agent_velocities = [[0.0, 0.0] for _ in range(self.num_agents)]
        self.objectives = [
            {
                "position": [
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1),
                ],
                "value": random.uniform(0.5, 2.0),
            }
            for _ in range(self.num_objectives)
        ]
        self.obstacles = self._generate_obstacles()
        self.hazards = self._generate_hazards()
        self.resources = self._generate_resources()
        self.current_step = 0
        return self.get_observation(), {}

    def get_observation(self) -> List[np.ndarray]:
        """
        Get observations for all agents. Observations include agent's own position,
        relative positions of other agents, objectives, obstacles, and hazards.

        Returns:
            List[np.ndarray]: A list of NumPy arrays, each representing an agent's observation.
        """
        observations = []
        for i, agent_pos in enumerate(self.agent_positions):
            obs = []
            # Own position
            obs.extend(agent_pos)

            # Relative positions of other agents
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:
                    obs.extend([other_pos[0] - agent_pos[0], other_pos[1] - agent_pos[1]])

            # Relative positions of objectives
            for obj in self.objectives:
                obs.extend([obj["position"][0] - agent_pos[0], obj["position"][1] - agent_pos[1], obj["value"]])

            # Relative positions of obstacles
            for obs_pos in self.obstacles:
                obs.extend([obs_pos[0] - agent_pos[0], obs_pos[1] - agent_pos[1]])

            # Relative positions of hazards
            for haz_pos in self.hazards:
                obs.extend([haz_pos[0] - agent_pos[0], haz_pos[1] - agent_pos[1]])

            # Add noise to observation
            observations.append(np.array(obs, dtype=np.float32) + np.random.normal(0, self.config.observation_noise, size=len(obs)))
        return observations

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Take a step in the environment given actions from all agents.

        Args:
            actions (List[np.ndarray]): A list of actions, one for each agent.
                                      For discrete actions, each element is an int (0:up, 1:down, 2:left, 3:right).
                                      For continuous actions (if physics_enabled), each element is a 2D force vector.

        Returns:
            Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
                - next_observations (List[np.ndarray]): New observations for all agents.
                - rewards (List[float]): Rewards for all agents.
                - dones (List[bool]): Done flags for all agents (True if episode finished for any agent).
                - info (Dict[str, Any]): Additional information about the step.
        """
        self.current_step += 1
        rewards = [0.0] * self.num_agents

        # Update agent positions based on actions and physics
        new_agent_positions = []
        for i, action in enumerate(actions):
            new_agent_positions.append(self._apply_physics(i, action))
        self.agent_positions = new_agent_positions

        # Check collisions and hazards
        collisions = self._check_collisions()
        in_hazard = self._check_hazards()

        # Compute rewards from objectives
        for i, agent_pos in enumerate(self.agent_positions):
            for obj in self.objectives:
                dist = np.linalg.norm(np.array(agent_pos) - np.array(obj["position"])) # type: ignore
                rewards[i] += obj["value"] / (1 + dist) # Reward inversely proportional to distance

            if collisions[i]:
                rewards[i] -= 5.0 # Penalty for collision
            if in_hazard[i]:
                rewards[i] -= 1.0 # Cost for being in hazard

        # Collect resources
        resource_rewards = self._collect_resources()
        for i in range(self.num_agents):
            rewards[i] += resource_rewards[i]

        # Dynamic environment changes
        self._update_objectives()

        # Check if episode is done
        dones = [self.current_step >= self.max_steps] * self.num_agents

        info = {"collisions": collisions, "in_hazard": in_hazard}
        return self.get_observation(), rewards, dones, info


class PartiallyObservableEnvironment:
    """
    Partially Observable Environment with limited visibility.

    Agents in this environment have a limited field of view, making full state
    knowledge unavailable and requiring strategies for dealing with uncertainty.

    Args:
        config (EnvironmentConfig): Configuration object for the environment settings.
    """

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.grid_size = config.size
        self.num_agents = config.num_agents
        self.partial_observability = config.partial_observability
        self.observation_noise = config.observation_noise

        self.agent_positions: List[List[int]] = []
        self.target_positions: List[List[int]] = []
        self.obstacle_positions: List[List[int]] = []
        self.enemy_positions: List[List[int]] = []
        self.visibility_radius = self.grid_size // 3  # Agents can only see within this radius
        self.current_step = 0
        self.max_steps = self.grid_size * 20

        self.reset()

    def _generate_targets(self) -> List[List[int]]:
        """
        Generate random target positions.

        Returns:
            List[List[int]]: A list of [x, y] coordinates for targets.
        """
        num_targets = self.grid_size // 4
        targets = []
        for _ in range(num_targets):
            targets.append([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        return targets

    def _generate_obstacles(self) -> List[List[int]]:
        """
        Generate random obstacle positions.

        Returns:
            List[List[int]]: A list of [x, y] coordinates for obstacles.
        """
        num_obstacles = self.grid_size // 3
        obstacles = []
        for _ in range(num_obstacles):
            obstacles.append([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        return obstacles

    def _generate_enemies(self) -> List[List[int]]:
        """
        Generate random enemy positions that move dynamically.

        Returns:
            List[List[int]]: A list of [x, y] coordinates for enemies.
        """
        num_enemies = self.grid_size // 5
        enemies = []
        for _ in range(num_enemies):
            enemies.append([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        return enemies

    def _get_visible_objects(self, agent_pos: List[int]) -> Dict[str, List[List[int]]]:
        """
        Determine which objects are visible to a given agent based on its visibility radius.

        Args:
            agent_pos (List[int]): The [x, y] position of the agent.

        Returns:
            Dict[str, List[List[int]]]: A dictionary containing visible targets, obstacles, and enemies.
        """
        visible_targets = []
        visible_obstacles = []
        visible_enemies = []

        for target_pos in self.target_positions:
            if self._is_in_sight(agent_pos, target_pos):
                visible_targets.append(target_pos)

        for obs_pos in self.obstacle_positions:
            if self._is_in_sight(agent_pos, obs_pos):
                visible_obstacles.append(obs_pos)

        for enemy_pos in self.enemy_positions:
            if self._is_in_sight(agent_pos, enemy_pos):
                visible_enemies.append(enemy_pos)

        return {
            "targets": visible_targets,
            "obstacles": visible_obstacles,
            "enemies": visible_enemies,
        }

    def _is_in_sight(self, agent_pos: List[int], object_pos: List[int]) -> bool:
        """
        Check if an object is within the agent's visibility radius.

        Args:
            agent_pos (List[int]): The [x, y] position of the agent.
            object_pos (List[int]): The [x, y] position of the object.

        Returns:
            bool: True if the object is in sight, False otherwise.
        """
        distance = math.sqrt((agent_pos[0] - object_pos[0])**2 + (agent_pos[1] - object_pos[1])**2)
        return distance <= self.visibility_radius

    def _update_enemies(self):
        """
        Update the positions of enemies randomly.
        """
        for i in range(len(self.enemy_positions)):
            move = random.choice([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
            new_pos = [
                (self.enemy_positions[i][0] + move[0]) % self.grid_size,
                (self.enemy_positions[i][1] + move[1]) % self.grid_size,
            ]
            self.enemy_positions[i] = new_pos

    def reset(self) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple[List[np.ndarray], Dict[str, Any]]: A tuple containing initial observations for all agents
                                                   and an info dictionary.
        """
        self.agent_positions = [
            [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            for _ in range(self.num_agents)
        ]
        self.target_positions = self._generate_targets()
        self.obstacle_positions = self._generate_obstacles()
        self.enemy_positions = self._generate_enemies()
        self.current_step = 0
        return self.get_observation(), {}

    def get_observation(self) -> List[np.ndarray]:
        """
        Get observations for all agents, respecting partial observability.

        Returns:
            List[np.ndarray]: A list of NumPy arrays, each representing an agent's observation.
        """
        observations = []
        for i, agent_pos in enumerate(self.agent_positions):
            obs = []
            # Own position
            obs.extend(agent_pos)

            if self.partial_observability:
                visible_objects = self._get_visible_objects(agent_pos)
                # Relative positions of visible targets
                for target_pos in visible_objects["targets"]:
                    obs.extend([target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1]])

                # Relative positions of visible obstacles
                for obs_pos in visible_objects["obstacles"]:
                    obs.extend([obs_pos[0] - agent_pos[0], obs_pos[1] - agent_pos[1]])

                # Relative positions of visible enemies
                for enemy_pos in visible_objects["enemies"]:
                    obs.extend([enemy_pos[0] - agent_pos[0], enemy_pos[1] - agent_pos[1]])
            else:
                # Full observability
                for target_pos in self.target_positions:
                    obs.extend([target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1]])
                for obs_pos in self.obstacle_positions:
                    obs.extend([obs_pos[0] - agent_pos[0], obs_pos[1] - agent_pos[1]])
                for enemy_pos in self.enemy_positions:
                    obs.extend([enemy_pos[0] - agent_pos[0], enemy_pos[1] - agent_pos[1]])

            # Add noise to observation
            observations.append(np.array(obs, dtype=np.float32) + np.random.normal(0, self.observation_noise, size=len(obs)))
        return observations

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Take a step in the environment given actions from all agents.

        Args:
            actions (List[int]): A list of integer actions, one for each agent (0:up, 1:down, 2:left, 3:right).

        Returns:
            Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
                - next_observations (List[np.ndarray]): New observations for all agents.
                - rewards (List[float]): Rewards for all agents.
                - dones (List[bool]): Done flags for all agents (True if episode finished for any agent).
                - info (Dict[str, Any]): Additional information about the step.
        """
        self.current_step += 1
        rewards = [0.0] * self.num_agents

        # Update agent positions
        new_agent_positions = []
        for i, action in enumerate(actions):
            current_pos = list(self.agent_positions[i]) # Ensure it's a list for modification
            if action == 0:  # Up
                current_pos[1] = (current_pos[1] - 1) % self.grid_size
            elif action == 1:  # Down
                current_pos[1] = (current_pos[1] + 1) % self.grid_size
            elif action == 2:  # Left
                current_pos[0] = (current_pos[0] - 1) % self.grid_size
            elif action == 3:  # Right
                current_pos[0] = (current_pos[0] + 1) % self.grid_size
            new_agent_positions.append(current_pos)
        self.agent_positions = new_agent_positions

        self._update_enemies() # Enemies move randomly

        # Compute rewards
        for i, agent_pos in enumerate(self.agent_positions):
            # Reward for reaching targets
            if agent_pos in self.target_positions:
                rewards[i] += 10.0
                self.target_positions.remove(agent_pos) # Remove collected target
                self.target_positions.extend(self._generate_targets()) # Respawn new target

            # Penalty for hitting obstacles
            if agent_pos in self.obstacle_positions:
                rewards[i] -= 5.0

            # Penalty for encountering enemies
            if agent_pos in self.enemy_positions:
                rewards[i] -= 3.0

        # Check if episode is done
        dones = [self.current_step >= self.max_steps] * self.num_agents

        info = {"targets_remaining": len(self.target_positions)}
        return self.get_observation(), rewards, dones, info

    def _is_valid_position(self, pos: List[int]) -> bool:
        """
        Check if a given position is within the grid boundaries.

        Args:
            pos (List[int]): The [x, y] position to check.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size


class ContinuousControlEnvironment:
    """
    Continuous Control Environment with realistic dynamics.

    This environment simulates a physics-based continuous control task, where
    agents exert continuous forces (actions) to navigate to targets.

    Args:
        config (EnvironmentConfig): Configuration object for the environment settings.
    """

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.size = config.size
        self.num_agents = config.num_agents
        self.action_noise = config.action_noise
        self.physics_enabled = True  # Always enabled for continuous control

        self.agent_positions: List[np.ndarray] = []  # Continuous positions
        self.agent_velocities: List[np.ndarray] = []
        self.agent_mass = 1.0
        self.targets: List[np.ndarray] = []
        self.obstacles: List[np.ndarray] = []

        self.current_step = 0
        self.max_steps = self.size * 50
        self.dt = 0.1  # Time step for physics simulation

        self.reset()

    def _generate_targets(self) -> List[np.ndarray]:
        """
        Generate random target positions in continuous space.

        Returns:
            List[np.ndarray]: A list of 2D NumPy arrays for target positions.
        """
        num_targets = self.size // 4
        targets = []
        for _ in range(num_targets):
            targets.append(np.random.uniform(0, self.size, size=2))
        return targets

    def _generate_obstacles(self) -> List[np.ndarray]:
        """
        Generate random obstacle positions in continuous space.

        Returns:
            List[np.ndarray]: A list of 2D NumPy arrays for obstacle positions.
        """
        num_obstacles = self.size // 3
        obstacles = []
        for _ in range(num_obstacles):
            obstacles.append(np.random.uniform(0, self.size, size=2))
        return obstacles

    def reset(self) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple[List[np.ndarray], Dict[str, Any]]: A tuple containing initial observations for all agents
                                                   and an info dictionary.
        """
        self.agent_positions = [np.random.uniform(0, self.size, size=2) for _ in range(self.num_agents)]
        self.agent_velocities = [np.zeros(2) for _ in range(self.num_agents)]
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()
        self.current_step = 0
        return self.get_observation(), {}

    def get_observation(self) -> List[np.ndarray]:
        """
        Get observations for all agents in continuous space.

        Observations include agent's own position, velocity, and relative positions
        of targets and obstacles.

        Returns:
            List[np.ndarray]: A list of NumPy arrays, each representing an agent's observation.
        """
        observations = []
        for i, agent_pos in enumerate(self.agent_positions):
            obs = []
            # Own position and velocity
            obs.extend(agent_pos.tolist())
            obs.extend(self.agent_velocities[i].tolist())

            # Relative positions of targets
            for target_pos in self.targets:
                obs.extend((target_pos - agent_pos).tolist())

            # Relative positions of obstacles
            for obs_pos in self.obstacles:
                obs.extend((obs_pos - agent_pos).tolist())

            # Add noise to observation
            observations.append(np.array(obs, dtype=np.float32) + np.random.normal(0, self.config.observation_noise, size=len(obs)))
        return observations

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Take a step in the environment given continuous actions (forces) from all agents.

        Args:
            actions (List[np.ndarray]): A list of 2D NumPy arrays, each representing a continuous
                                      force vector (action) for an agent.

        Returns:
            Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
                - next_observations (List[np.ndarray]): New observations for all agents.
                - rewards (List[float]): Rewards for all agents.
                - dones (List[bool]): Done flags for all agents (True if episode finished for any agent).
                - info (Dict[str, Any]): Additional information about the step.
        """
        self.current_step += 1
        rewards = [0.0] * self.num_agents
        dones = [self.current_step >= self.max_steps] * self.num_agents

        new_agent_positions = []
        for i, action in enumerate(actions):
            # Apply action noise
            action_with_noise = action + np.random.normal(0, self.action_noise, size=action.shape)

            # Apply physics: F = ma, a = F/m, v = v + a*dt, p = p + v*dt
            acceleration = action_with_noise / self.agent_mass
            self.agent_velocities[i] = self.agent_velocities[i] + acceleration * self.dt

            # Update position
            new_pos = self.agent_positions[i] + self.agent_velocities[i] * self.dt

            # Boundary conditions (wrap around)
            new_pos = np.mod(new_pos, self.size)
            new_agent_positions.append(new_pos)
        self.agent_positions = new_agent_positions

        # Compute rewards
        for i, agent_pos in enumerate(self.agent_positions):
            # Reward for being close to targets
            for target_pos in self.targets:
                distance_to_target = np.linalg.norm(agent_pos - target_pos)
                rewards[i] += 1.0 / (1.0 + distance_to_target) # Inverse distance reward

            # Penalty for being close to obstacles
            for obs_pos in self.obstacles:
                distance_to_obstacle = np.linalg.norm(agent_pos - obs_pos)
                if distance_to_obstacle < 0.5: # Collision radius
                    rewards[i] -= 5.0
                    dones[i] = True # End episode on collision

        info = {}
        return self.get_observation(), rewards, dones, info


class AdversarialEnvironment:
    """
    Adversarial Environment with adaptive opponents.

    This environment includes an adaptive adversarial agent that attempts to
    disrupt the main agent's performance, suitable for testing robust RL algorithms.

    Args:
        config (EnvironmentConfig): Configuration object for the environment settings.
    """

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.grid_size = config.size
        self.num_agents = config.num_agents  # Main agents
        self.adversarial_mode = config.adversarial_mode
        self.observation_noise = config.observation_noise

        self.agent_positions: List[List[int]] = []
        self.adversary_position: Optional[List[int]] = None # Single adversary for simplicity
        self.targets: List[List[int]] = []
        self.obstacles: List[List[int]] = []

        self.current_step = 0
        self.max_steps = self.grid_size * 20
        self.adversary_strength = 0.5 # How much the adversary can influence

        self.reset()

    def _generate_targets(self) -> List[List[int]]:
        """
        Generate random target positions.

        Returns:
            List[List[int]]: A list of [x, y] coordinates for targets.
        """
        num_targets = self.grid_size // 4
        targets = []
        for _ in range(num_targets):
            targets.append([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        return targets

    def _generate_obstacles(self) -> List[List[int]]:
        """
        Generate random obstacle positions.

        Returns:
            List[List[int]]: A list of [x, y] coordinates for obstacles.
        """
        num_obstacles = self.grid_size // 3
        obstacles = []
        for _ in range(num_obstacles):
            obstacles.append([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        return obstacles

    def _update_opponent_strategy(self):
        """
        Update the adversarial opponent's strategy.

        This could involve adapting its policy based on the main agents' performance,
        e.g., moving closer to agents with high rewards, or blocking paths.
        For now, it moves randomly towards the closest agent.
        """
        if self.adversary_position is None or not self.adversarial_mode:
            return

        # Simple adversarial strategy: move towards the closest agent
        if self.agent_positions:
            closest_agent_pos = min(
                self.agent_positions, key=lambda p: np.linalg.norm(np.array(p) - np.array(self.adversary_position)) # type: ignore
            )
            adv_move = [0, 0]
            if closest_agent_pos[0] > self.adversary_position[0]:
                adv_move[0] = 1
            elif closest_agent_pos[0] < self.adversary_position[0]:
                adv_move[0] = -1
            if closest_agent_pos[1] > self.adversary_position[1]:
                adv_move[1] = 1
            elif closest_agent_pos[1] < self.adversary_position[1]:
                adv_move[1] = -1

            self.adversary_position = [
                (self.adversary_position[0] + adv_move[0]) % self.grid_size,
                (self.adversary_position[1] + adv_move[1]) % self.grid_size,
            ]

    def _get_opponent_action(self) -> List[int]:
        """
        Get the adversarial opponent's action.
        For this simplified environment, the opponent's action is determined by its internal strategy.

        Returns:
            List[int]: The [dx, dy] movement chosen by the opponent.
        """
        # Placeholder for more complex adversarial policies
        if self.adversary_position is None:
            return [0, 0]
        # The _update_opponent_strategy already updates the position, so this just returns a dummy action.
        return [0, 0]

    def reset(self) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Returns:
            Tuple[List[np.ndarray], Dict[str, Any]]: A tuple containing initial observations for all main agents
                                                   and an info dictionary.
        """
        self.agent_positions = [
            [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            for _ in range(self.num_agents)
        ]
        if self.adversarial_mode:
            self.adversary_position = [
                random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            ]
        else:
            self.adversary_position = None
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()
        self.current_step = 0
        return self.get_observation(), {}

    def get_observation(self) -> List[np.ndarray]:
        """
        Get observations for all main agents.

        Observations include agent's own position, relative positions of other agents,
        targets, obstacles, and if in adversarial mode, the adversary's relative position.

        Returns:
            List[np.ndarray]: A list of NumPy arrays, each representing a main agent's observation.
        """
        observations = []
        for i, agent_pos in enumerate(self.agent_positions):
            obs = []
            # Own position
            obs.extend(agent_pos)

            # Relative positions of other agents
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:
                    obs.extend([other_pos[0] - agent_pos[0], other_pos[1] - agent_pos[1]])

            # Relative positions of targets
            for target_pos in self.targets:
                obs.extend([target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1]])

            # Relative positions of obstacles
            for obs_pos in self.obstacles:
                obs.extend([obs_pos[0] - agent_pos[0], obs_pos[1] - agent_pos[1]])

            # Adversary position
            if self.adversarial_mode and self.adversary_position:
                obs.extend([self.adversary_position[0] - agent_pos[0], self.adversary_position[1] - agent_pos[1]])

            # Add noise to observation
            observations.append(np.array(obs, dtype=np.float32) + np.random.normal(0, self.observation_noise, size=len(obs)))
        return observations

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Take a step in the environment given actions from all main agents.

        Args:
            actions (List[int]): A list of integer actions, one for each main agent (0:up, 1:down, 2:left, 3:right).

        Returns:
            Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
                - next_observations (List[np.ndarray]): New observations for all main agents.
                - rewards (List[float]): Rewards for all main agents.
                - dones (List[bool]): Done flags for all main agents (True if episode finished for any agent).
                - info (Dict[str, Any]): Additional information about the step.
        """
        self.current_step += 1
        rewards = [0.0] * self.num_agents

        # Update main agent positions
        new_agent_positions = []
        for i, action in enumerate(actions):
            new_pos = self._apply_action(self.agent_positions[i], action)
            new_agent_positions.append(new_pos)
        self.agent_positions = new_agent_positions

        # Update adversary position if in adversarial mode
        if self.adversarial_mode:
            self._update_opponent_strategy()

        # Compute rewards
        for i, agent_pos in enumerate(self.agent_positions):
            # Reward for reaching targets
            if agent_pos in self.targets:
                rewards[i] += 10.0
                self.targets.remove(agent_pos) # Remove collected target
                self.targets.extend(self._generate_targets()) # Respawn new target

            # Penalty for hitting obstacles
            if agent_pos in self.obstacles:
                rewards[i] -= 5.0

            # Penalty for being close to adversary
            if self.adversarial_mode and self.adversary_position:
                dist_to_adversary = np.linalg.norm(np.array(agent_pos) - np.array(self.adversary_position)) # type: ignore
                if dist_to_adversary < 1.5: # Close proximity penalty
                    rewards[i] -= 3.0 * (1.5 - dist_to_adversary) * self.adversary_strength

        # Check if episode is done
        dones = [self.current_step >= self.max_steps] * self.num_agents

        info = {"targets_remaining": len(self.targets)}
        if self.adversarial_mode and self.adversary_position:
            info["adversary_pos"] = self.adversary_position
        return self.get_observation(), rewards, dones, info

    def _apply_action(self, pos: List[int], action: int) -> List[int]:
        """
        Apply a discrete action to a given position.

        Args:
            pos (List[int]): The current [x, y] position.
            action (int): The discrete action (0:up, 1:down, 2:left, 3:right).

        Returns:
            List[int]: The new [x, y] position after applying the action.
        """
        new_pos = list(pos)
        if action == 0:  # Up
            new_pos[1] = (new_pos[1] - 1) % self.grid_size
        elif action == 1:  # Down
            new_pos[1] = (new_pos[1] + 1) % self.grid_size
        elif action == 2:  # Left
            new_pos[0] = (new_pos[0] - 1) % self.grid_size
        elif action == 3:  # Right
            new_pos[0] = (new_pos[0] + 1) % self.grid_size
        return new_pos
