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
    """Configuration for complex environments."""

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
    """Dynamic Multi-Objective Environment with changing goals."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.size = config.size
        self.num_objectives = config.num_objectives
        self.dynamic_changes = config.dynamic_changes

        # Dynamic objectives
        self.objectives = []
        self.objective_weights = np.random.dirichlet(np.ones(config.num_objectives))
        self.objective_change_frequency = 50
        self.step_count = 0

        # Agent state
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.agent_velocity = np.array([0, 0], dtype=np.float32)

        # Environment state
        self.obstacles = self._generate_obstacles()
        self.resources = self._generate_resources()
        self.hazards = self._generate_hazards()

        # Physics parameters
        self.friction = 0.9
        self.max_velocity = 2.0

        # Observation space
        self.observation_space = (
            2 + 2 + self.num_objectives * 2 + 10
        )  # pos + vel + objectives + local info
        self.action_space = 4  # up, down, left, right

    def _generate_obstacles(self):
        """Generate dynamic obstacles."""
        obstacles = []
        for _ in range(self.size // 3):
            pos = np.random.randint(1, self.size - 1, 2)
            obstacles.append(pos)
        return obstacles

    def _generate_resources(self):
        """Generate resources for objectives."""
        resources = []
        for _ in range(self.num_objectives * 3):
            pos = np.random.randint(0, self.size, 2)
            resource_type = np.random.randint(0, self.num_objectives)
            value = np.random.uniform(0.5, 2.0)
            resources.append({"pos": pos, "type": resource_type, "value": value})
        return resources

    def _generate_hazards(self):
        """Generate hazardous areas."""
        hazards = []
        for _ in range(self.size // 4):
            pos = np.random.randint(0, self.size, 2)
            radius = np.random.uniform(0.5, 1.5)
            damage = np.random.uniform(0.1, 0.5)
            hazards.append({"pos": pos, "radius": radius, "damage": damage})
        return hazards

    def _update_objectives(self):
        """Update dynamic objectives."""
        if (
            self.dynamic_changes
            and self.step_count % self.objective_change_frequency == 0
        ):
            # Change objective weights
            self.objective_weights = np.random.dirichlet(np.ones(self.num_objectives))

            # Move some objectives
            for i in range(len(self.objectives)):
                if np.random.random() < 0.3:  # 30% chance to move
                    self.objectives[i] = np.random.randint(0, self.size, 2)

    def _apply_physics(self, action):
        """Apply realistic physics to agent movement."""
        # Convert action to force
        force = np.array([0.0, 0.0])
        if action == 0:  # up
            force[1] = 1.0
        elif action == 1:  # down
            force[1] = -1.0
        elif action == 2:  # left
            force[0] = -1.0
        elif action == 3:  # right
            force[0] = 1.0

        # Apply force to velocity
        self.agent_velocity += force * 0.1

        # Apply friction
        self.agent_velocity *= self.friction

        # Limit velocity
        velocity_magnitude = np.linalg.norm(self.agent_velocity)
        if velocity_magnitude > self.max_velocity:
            self.agent_velocity = (
                self.agent_velocity / velocity_magnitude * self.max_velocity
            )

        # Update position
        self.agent_pos = self.agent_pos.astype(np.float32) + self.agent_velocity.astype(
            np.float32
        )

        # Boundary constraints
        self.agent_pos = np.clip(self.agent_pos, 0.0, float(self.size - 1))

    def _check_collisions(self):
        """Check for collisions with obstacles."""
        collision_penalty = 0
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.agent_pos - obstacle)
            if distance < 0.5:  # Collision threshold
                collision_penalty += 1.0
                # Push agent away from obstacle
                direction = self.agent_pos - obstacle
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    self.agent_pos = obstacle + direction * 0.6
        return collision_penalty

    def _check_hazards(self):
        """Check for hazard damage."""
        damage = 0
        for hazard in self.hazards:
            distance = np.linalg.norm(self.agent_pos - hazard["pos"])
            if distance < hazard["radius"]:
                damage += hazard["damage"] * (1 - distance / hazard["radius"])
        return damage

    def _collect_resources(self):
        """Collect nearby resources."""
        collected_value = np.zeros(self.num_objectives)
        resources_to_remove = []

        for i, resource in enumerate(self.resources):
            distance = np.linalg.norm(self.agent_pos - resource["pos"])
            if distance < 0.5:  # Collection threshold
                collected_value[resource["type"]] += resource["value"]
                resources_to_remove.append(i)

        # Remove collected resources
        for i in reversed(resources_to_remove):
            self.resources.pop(i)

        return collected_value

    def reset(self):
        """Reset environment."""
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.agent_velocity = np.array([0, 0], dtype=np.float32)
        self.step_count = 0

        # Regenerate dynamic elements
        self.obstacles = self._generate_obstacles()
        self.resources = self._generate_resources()
        self.hazards = self._generate_hazards()

        # Initialize objectives
        self.objectives = []
        for _ in range(self.num_objectives):
            self.objectives.append(np.random.randint(0, self.size, 2))

        return self.get_observation()

    def get_observation(self):
        """Get current observation."""
        obs = []

        # Agent position and velocity
        obs.extend(self.agent_pos / self.size)  # Normalized position
        obs.extend(self.agent_velocity / self.max_velocity)  # Normalized velocity

        # Objective information
        for objective in self.objectives:
            rel_pos = (objective - self.agent_pos) / self.size
            obs.extend(rel_pos)

        # Local environment information
        local_info = self._get_local_info()
        obs.extend(local_info)

        return np.array(obs, dtype=np.float32)

    def _get_local_info(self):
        """Get local environment information."""
        local_info = []

        # Distance to nearest obstacle
        min_obstacle_dist = float("inf")
        for obstacle in self.obstacles:
            dist = np.linalg.norm(self.agent_pos - obstacle)
            min_obstacle_dist = min(min_obstacle_dist, dist)
        local_info.append(min_obstacle_dist / self.size)

        # Distance to nearest resource
        min_resource_dist = float("inf")
        for resource in self.resources:
            dist = np.linalg.norm(self.agent_pos - resource["pos"])
            min_resource_dist = min(min_resource_dist, dist)
        local_info.append(min_resource_dist / self.size)

        # Distance to nearest hazard
        min_hazard_dist = float("inf")
        for hazard in self.hazards:
            dist = np.linalg.norm(self.agent_pos - hazard["pos"])
            min_hazard_dist = min(min_hazard_dist, dist)
        local_info.append(min_hazard_dist / self.size)

        # Resource density in local area
        local_resources = 0
        for resource in self.resources:
            if np.linalg.norm(self.agent_pos - resource["pos"]) < 2.0:
                local_resources += 1
        local_info.append(local_resources / 10.0)

        # Fill remaining space
        while len(local_info) < 10:
            local_info.append(0.0)

        return local_info[:10]

    def step(self, action):
        """Take step in environment."""
        self.step_count += 1

        # Update dynamic objectives
        self._update_objectives()

        # Apply physics
        self._apply_physics(action)

        # Check collisions and hazards
        collision_penalty = self._check_collisions()
        hazard_damage = self._check_hazards()

        # Collect resources
        collected_value = self._collect_resources()

        # Compute multi-objective reward
        reward = np.sum(collected_value * self.objective_weights)
        reward -= collision_penalty
        reward -= hazard_damage

        # Check termination
        done = self.step_count >= 200 or np.sum(collected_value) > 10

        info = {
            "collected_value": collected_value,
            "objective_weights": self.objective_weights,
            "collision_penalty": collision_penalty,
            "hazard_damage": hazard_damage,
            "step_count": self.step_count,
        }

        return self.get_observation(), reward, done, info


class PartiallyObservableEnvironment:
    """Partially Observable Environment with limited visibility."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.size = config.size
        self.visibility_radius = 3.0
        self.observation_noise = config.observation_noise

        # Agent state
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.agent_orientation = 0.0  # Angle in radians

        # Environment state
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()
        self.enemies = self._generate_enemies()

        # Observation space (limited by visibility)
        self.observation_space = 2 + 8 + 4  # pos + local_map + orientation_info
        self.action_space = 5  # forward, left, right, turn_left, turn_right

    def _generate_targets(self):
        """Generate targets to collect."""
        targets = []
        for _ in range(5):
            pos = np.random.randint(0, self.size, 2)
            value = np.random.uniform(0.5, 2.0)
            targets.append({"pos": pos, "value": value, "collected": False})
        return targets

    def _generate_obstacles(self):
        """Generate obstacles."""
        obstacles = []
        for _ in range(self.size // 2):
            pos = np.random.randint(0, self.size, 2)
            obstacles.append(pos)
        return obstacles

    def _generate_enemies(self):
        """Generate enemy agents."""
        enemies = []
        for _ in range(3):
            pos = np.random.randint(0, self.size, 2)
            speed = np.random.uniform(0.5, 1.5)
            enemies.append(
                {
                    "pos": pos,
                    "speed": speed,
                    "direction": np.random.uniform(0, 2 * np.pi),
                }
            )
        return enemies

    def _get_visible_objects(self):
        """Get objects within visibility radius."""
        visible_objects = {"targets": [], "obstacles": [], "enemies": []}

        for target in self.targets:
            if not target["collected"]:
                distance = np.linalg.norm(self.agent_pos - target["pos"])
                if distance <= self.visibility_radius:
                    # Check if object is in field of view
                    angle_to_target = np.arctan2(
                        target["pos"][1] - self.agent_pos[1],
                        target["pos"][0] - self.agent_pos[0],
                    )
                    angle_diff = abs(angle_to_target - self.agent_orientation)
                    if angle_diff <= np.pi / 2:  # 90-degree field of view
                        visible_objects["targets"].append(
                            {
                                "pos": target["pos"],
                                "value": target["value"],
                                "distance": distance,
                            }
                        )

        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.agent_pos - obstacle)
            if distance <= self.visibility_radius:
                angle_to_obstacle = np.arctan2(
                    obstacle[1] - self.agent_pos[1], obstacle[0] - self.agent_pos[0]
                )
                angle_diff = abs(angle_to_obstacle - self.agent_orientation)
                if angle_diff <= np.pi / 2:
                    visible_objects["obstacles"].append(
                        {"pos": obstacle, "distance": distance}
                    )

        for enemy in self.enemies:
            distance = np.linalg.norm(self.agent_pos - enemy["pos"])
            if distance <= self.visibility_radius:
                angle_to_enemy = np.arctan2(
                    enemy["pos"][1] - self.agent_pos[1],
                    enemy["pos"][0] - self.agent_pos[0],
                )
                angle_diff = abs(angle_to_enemy - self.agent_orientation)
                if angle_diff <= np.pi / 2:
                    visible_objects["enemies"].append(
                        {
                            "pos": enemy["pos"],
                            "distance": distance,
                            "speed": enemy["speed"],
                        }
                    )

        return visible_objects

    def _update_enemies(self):
        """Update enemy positions."""
        for enemy in self.enemies:
            # Simple random movement
            enemy["direction"] += np.random.uniform(-0.5, 0.5)
            enemy["pos"] += enemy["speed"] * np.array(
                [np.cos(enemy["direction"]), np.sin(enemy["direction"])]
            )

            # Keep enemies within bounds
            enemy["pos"] = np.clip(enemy["pos"], 0, self.size - 1)

    def reset(self):
        """Reset environment."""
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.agent_orientation = 0.0

        # Regenerate environment
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()
        self.enemies = self._generate_enemies()

        return self.get_observation()

    def get_observation(self):
        """Get partial observation."""
        obs = []

        # Agent position and orientation
        obs.extend(self.agent_pos / self.size)
        obs.append(self.agent_orientation / (2 * np.pi))  # Normalized orientation

        # Visible objects
        visible_objects = self._get_visible_objects()

        # Local map (8 directions)
        local_map = [0.0] * 8
        for target in visible_objects["targets"]:
            angle = np.arctan2(
                target["pos"][1] - self.agent_pos[1],
                target["pos"][0] - self.agent_pos[0],
            )
            direction = int((angle + np.pi) / (2 * np.pi) * 8) % 8
            local_map[direction] = target["value"]

        for obstacle in visible_objects["obstacles"]:
            angle = np.arctan2(
                obstacle["pos"][1] - self.agent_pos[1],
                obstacle["pos"][0] - self.agent_pos[0],
            )
            direction = int((angle + np.pi) / (2 * np.pi) * 8) % 8
            local_map[direction] = -1.0  # Obstacle indicator

        obs.extend(local_map)

        # Orientation information
        obs.append(np.cos(self.agent_orientation))
        obs.append(np.sin(self.agent_orientation))
        obs.append(0.0)  # Placeholder
        obs.append(0.0)  # Placeholder

        # Add observation noise
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, len(obs))
            obs = np.array(obs) + noise

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Take step in environment."""
        # Update enemies
        self._update_enemies()

        # Apply action
        if action == 0:  # forward
            new_pos = self.agent_pos + 0.5 * np.array(
                [np.cos(self.agent_orientation), np.sin(self.agent_orientation)]
            )
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
        elif action == 1:  # left
            new_pos = self.agent_pos + 0.5 * np.array(
                [
                    np.cos(self.agent_orientation + np.pi / 2),
                    np.sin(self.agent_orientation + np.pi / 2),
                ]
            )
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
        elif action == 2:  # right
            new_pos = self.agent_pos + 0.5 * np.array(
                [
                    np.cos(self.agent_orientation - np.pi / 2),
                    np.sin(self.agent_orientation - np.pi / 2),
                ]
            )
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
        elif action == 3:  # turn_left
            self.agent_orientation += np.pi / 8
        elif action == 4:  # turn_right
            self.agent_orientation -= np.pi / 8

        # Keep agent within bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        # Check for target collection
        reward = 0
        for target in self.targets:
            if not target["collected"]:
                distance = np.linalg.norm(self.agent_pos - target["pos"])
                if distance < 0.5:
                    reward += target["value"]
                    target["collected"] = True

        # Check for enemy collision
        penalty = 0
        for enemy in self.enemies:
            distance = np.linalg.norm(self.agent_pos - enemy["pos"])
            if distance < 0.5:
                penalty += 1.0

        # Check termination
        done = all(target["collected"] for target in self.targets) or penalty > 0

        info = {
            "targets_collected": sum(target["collected"] for target in self.targets),
            "total_targets": len(self.targets),
            "enemy_collision": penalty > 0,
        }

        return self.get_observation(), reward - penalty, done, info

    def _is_valid_position(self, pos):
        """Check if position is valid (not in obstacle)."""
        for obstacle in self.obstacles:
            if np.linalg.norm(pos - obstacle) < 0.5:
                return False
        return True


class ContinuousControlEnvironment:
    """Continuous Control Environment with realistic dynamics."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.size = config.size

        # Agent state (continuous)
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_angle = 0.0
        self.agent_angular_velocity = 0.0

        # Physics parameters
        self.mass = 1.0
        self.moment_of_inertia = 0.1
        self.drag_coefficient = 0.1
        self.max_force = 2.0
        self.max_torque = 1.0

        # Environment
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()

        # Action space (continuous force and torque)
        self.action_space = 3  # force_x, force_y, torque
        self.observation_space = (
            8  # pos, vel, angle, angular_vel, target_rel_pos, obstacle_dist
        )

    def _generate_targets(self):
        """Generate targets."""
        targets = []
        for _ in range(3):
            pos = np.random.uniform(1, self.size - 1, 2)
            targets.append({"pos": pos, "collected": False})
        return targets

    def _generate_obstacles(self):
        """Generate obstacles."""
        obstacles = []
        for _ in range(5):
            pos = np.random.uniform(0, self.size, 2)
            radius = np.random.uniform(0.3, 0.8)
            obstacles.append({"pos": pos, "radius": radius})
        return obstacles

    def reset(self):
        """Reset environment."""
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_angle = 0.0
        self.agent_angular_velocity = 0.0

        # Regenerate environment
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()

        return self.get_observation()

    def get_observation(self):
        """Get observation."""
        obs = []

        # Agent state
        obs.extend(self.agent_pos / self.size)
        obs.extend(self.agent_velocity / 5.0)  # Normalized velocity
        obs.append(self.agent_angle / (2 * np.pi))
        obs.append(self.agent_angular_velocity / 5.0)

        # Target information
        if self.targets:
            nearest_target = min(
                self.targets, key=lambda t: np.linalg.norm(self.agent_pos - t["pos"])
            )
            rel_pos = (nearest_target["pos"] - self.agent_pos) / self.size
            obs.extend(rel_pos)
        else:
            obs.extend([0.0, 0.0])

        # Obstacle information
        if self.obstacles:
            nearest_obstacle = min(
                self.obstacles, key=lambda o: np.linalg.norm(self.agent_pos - o["pos"])
            )
            distance = (
                np.linalg.norm(self.agent_pos - nearest_obstacle["pos"]) / self.size
            )
            obs.append(distance)
        else:
            obs.append(1.0)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Take step with continuous control."""
        # Parse action
        force_x = np.clip(action[0], -self.max_force, self.max_force)
        force_y = np.clip(action[1], -self.max_force, self.max_force)
        torque = np.clip(action[2], -self.max_torque, self.max_torque)

        # Apply forces
        force = np.array([force_x, force_y])
        acceleration = force / self.mass

        # Apply torque
        angular_acceleration = torque / self.moment_of_inertia

        # Update velocity
        self.agent_velocity += acceleration * 0.1
        self.agent_angular_velocity += angular_acceleration * 0.1

        # Apply drag
        self.agent_velocity *= 1 - self.drag_coefficient
        self.agent_angular_velocity *= 1 - self.drag_coefficient

        # Update position
        self.agent_pos += self.agent_velocity * 0.1
        self.agent_angle += self.agent_angular_velocity * 0.1

        # Keep angle in [0, 2π]
        self.agent_angle = self.agent_angle % (2 * np.pi)

        # Boundary constraints
        self.agent_pos = np.clip(self.agent_pos, 0, self.size)

        # Check collisions
        collision_penalty = 0
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.agent_pos - obstacle["pos"])
            if distance < obstacle["radius"]:
                collision_penalty += 1.0
                # Push agent away
                direction = self.agent_pos - obstacle["pos"]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    self.agent_pos = obstacle["pos"] + direction * obstacle["radius"]

        # Check target collection
        reward = 0
        for target in self.targets:
            if not target["collected"]:
                distance = np.linalg.norm(self.agent_pos - target["pos"])
                if distance < 0.3:
                    reward += 1.0
                    target["collected"] = True

        # Distance-based reward
        if self.targets:
            nearest_target = min(
                self.targets, key=lambda t: np.linalg.norm(self.agent_pos - t["pos"])
            )
            distance_reward = (
                -np.linalg.norm(self.agent_pos - nearest_target["pos"]) / self.size
            )
            reward += distance_reward * 0.1

        # Check termination
        done = (
            all(target["collected"] for target in self.targets) or collision_penalty > 0
        )

        info = {
            "targets_collected": sum(target["collected"] for target in self.targets),
            "collision_penalty": collision_penalty,
        }

        return self.get_observation(), reward - collision_penalty, done, info


class AdversarialEnvironment:
    """Adversarial Environment with adaptive opponents."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.size = config.size
        self.adversarial_mode = config.adversarial_mode

        # Agent state
        self.agent_pos = np.array([0, 0], dtype=np.float32)

        # Adversarial opponent
        self.opponent_pos = np.array([self.size - 1, self.size - 1], dtype=np.float32)
        self.opponent_strategy = "random"  # random, aggressive, defensive
        self.opponent_adaptation_rate = 0.1

        # Environment
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()

        # Strategy tracking
        self.agent_strategy_history = []
        self.opponent_performance = {"random": 0, "aggressive": 0, "defensive": 0}

        self.observation_space = (
            2 + 2 + 2 + 3
        )  # agent_pos + opponent_pos + target_pos + strategy_info
        self.action_space = 4

    def _generate_targets(self):
        """Generate targets."""
        targets = []
        for _ in range(3):
            pos = np.random.randint(0, self.size, 2)
            targets.append({"pos": pos, "collected": False, "collector": None})
        return targets

    def _generate_obstacles(self):
        """Generate obstacles."""
        obstacles = []
        for _ in range(self.size // 3):
            pos = np.random.randint(0, self.size, 2)
            obstacles.append(pos)
        return obstacles

    def _update_opponent_strategy(self):
        """Update opponent strategy based on agent behavior."""
        if not self.agent_strategy_history:
            return

        # Analyze agent strategy
        recent_strategies = self.agent_strategy_history[-10:]
        if len(recent_strategies) >= 5:
            # Determine agent strategy
            if np.mean(recent_strategies) > 0.6:
                agent_strategy = "aggressive"
            elif np.mean(recent_strategies) < 0.4:
                agent_strategy = "defensive"
            else:
                agent_strategy = "random"

            # Adapt opponent strategy
            if agent_strategy == "aggressive":
                self.opponent_strategy = "defensive"
            elif agent_strategy == "defensive":
                self.opponent_strategy = "aggressive"
            else:
                self.opponent_strategy = "random"

    def _get_opponent_action(self):
        """Get opponent action based on current strategy."""
        if self.opponent_strategy == "aggressive":
            # Move towards agent
            direction = self.agent_pos - self.opponent_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                action = np.argmax(np.abs(direction))
                if direction[action] < 0:
                    action += 2  # Convert to action space
        elif self.opponent_strategy == "defensive":
            # Move towards targets
            if self.targets:
                nearest_target = min(
                    self.targets,
                    key=lambda t: np.linalg.norm(self.opponent_pos - t["pos"]),
                )
                direction = nearest_target["pos"] - self.opponent_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    action = np.argmax(np.abs(direction))
                    if direction[action] < 0:
                        action += 2
            else:
                action = np.random.randint(0, 4)
        else:  # random
            action = np.random.randint(0, 4)

        return action

    def reset(self):
        """Reset environment."""
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.opponent_pos = np.array([self.size - 1, self.size - 1], dtype=np.float32)
        self.opponent_strategy = "random"

        # Regenerate environment
        self.targets = self._generate_targets()
        self.obstacles = self._generate_obstacles()

        # Reset strategy tracking
        self.agent_strategy_history = []
        self.opponent_performance = {"random": 0, "aggressive": 0, "defensive": 0}

        return self.get_observation()

    def get_observation(self):
        """Get observation."""
        obs = []

        # Agent position
        obs.extend(self.agent_pos / self.size)

        # Opponent position
        obs.extend(self.opponent_pos / self.size)

        # Nearest target position
        if self.targets:
            nearest_target = min(
                self.targets, key=lambda t: np.linalg.norm(self.agent_pos - t["pos"])
            )
            obs.extend(nearest_target["pos"] / self.size)
        else:
            obs.extend([0.0, 0.0])

        # Strategy information
        strategy_vector = [0.0, 0.0, 0.0]
        if self.opponent_strategy == "random":
            strategy_vector[0] = 1.0
        elif self.opponent_strategy == "aggressive":
            strategy_vector[1] = 1.0
        else:  # defensive
            strategy_vector[2] = 1.0
        obs.extend(strategy_vector)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Take step in adversarial environment."""
        # Update opponent strategy
        self._update_opponent_strategy()

        # Get opponent action
        opponent_action = self._get_opponent_action()

        # Apply agent action
        self._apply_action(self.agent_pos, action)

        # Apply opponent action
        self._apply_action(self.opponent_pos, opponent_action)

        # Check target collection
        reward = 0
        for target in self.targets:
            if not target["collected"]:
                agent_distance = np.linalg.norm(self.agent_pos - target["pos"])
                opponent_distance = np.linalg.norm(self.opponent_pos - target["pos"])

                if agent_distance < 0.5 and opponent_distance >= 0.5:
                    reward += 1.0
                    target["collected"] = True
                    target["collector"] = "agent"
                elif opponent_distance < 0.5 and agent_distance >= 0.5:
                    reward -= 0.5
                    target["collected"] = True
                    target["collector"] = "opponent"

        # Track strategy
        self.agent_strategy_history.append(action / 4.0)  # Normalize action

        # Check termination
        done = all(target["collected"] for target in self.targets)

        info = {
            "targets_collected": sum(
                1 for t in self.targets if t["collector"] == "agent"
            ),
            "opponent_strategy": self.opponent_strategy,
            "opponent_targets": sum(
                1 for t in self.targets if t["collector"] == "opponent"
            ),
        }

        return self.get_observation(), reward, done, info

    def _apply_action(self, pos, action):
        """Apply action to position."""
        new_pos = pos.copy()

        if action == 0 and pos[1] < self.size - 1:  # up
            new_pos[1] += 1
        elif action == 1 and pos[1] > 0:  # down
            new_pos[1] -= 1
        elif action == 2 and pos[0] > 0:  # left
            new_pos[0] -= 1
        elif action == 3 and pos[0] < self.size - 1:  # right
            new_pos[0] += 1

        # Check obstacle collision
        if new_pos.tolist() not in [obs.tolist() for obs in self.obstacles]:
            pos[:] = new_pos
