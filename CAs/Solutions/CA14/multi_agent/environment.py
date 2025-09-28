"""
Multi-Agent Environment for Multi-Agent Reinforcement Learning

This module provides environments for testing multi-agent RL algorithms.
"""

import numpy as np


class MultiAgentEnvironment:
    """Multi-agent environment for MARL demonstration."""

    def __init__(self, grid_size=8, num_agents=4, num_targets=3):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.max_episode_steps = 100

        # Initialize agent and target positions
        self.reset()

        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_space = 5
        self.observation_space = (
            2 + 2 * num_agents + 2 * num_targets
        )  # pos + other_agents + targets

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0

        # Random agent positions
        self.agent_positions = []
        for _ in range(self.num_agents):
            while True:
                pos = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size),
                ]
                if pos not in self.agent_positions:
                    self.agent_positions.append(pos)
                    break

        # Random target positions
        self.target_positions = []
        for _ in range(self.num_targets):
            while True:
                pos = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size),
                ]
                if pos not in self.agent_positions and pos not in self.target_positions:
                    self.target_positions.append(pos)
                    break

        self.targets_collected = [False] * self.num_targets
        return self.get_observations()

    def get_observations(self):
        """Get observations for all agents."""
        observations = []

        for i in range(self.num_agents):
            obs = []

            # Agent's own position (normalized)
            obs.extend(
                [
                    self.agent_positions[i][0] / self.grid_size,
                    self.agent_positions[i][1] / self.grid_size,
                ]
            )

            # Other agents' positions (relative)
            for j in range(self.num_agents):
                if i != j:
                    rel_pos = [
                        (self.agent_positions[j][0] - self.agent_positions[i][0])
                        / self.grid_size,
                        (self.agent_positions[j][1] - self.agent_positions[i][1])
                        / self.grid_size,
                    ]
                    obs.extend(rel_pos)

            # Target positions (relative) and collection status
            for k, target_pos in enumerate(self.target_positions):
                if not self.targets_collected[k]:
                    rel_pos = [
                        (target_pos[0] - self.agent_positions[i][0]) / self.grid_size,
                        (target_pos[1] - self.agent_positions[i][1]) / self.grid_size,
                    ]
                    obs.extend(rel_pos)
                else:
                    obs.extend([0.0, 0.0])  # Target collected

            observations.append(np.array(obs, dtype=np.float32))

        return observations

    def step(self, actions):
        """Execute joint action and return results."""
        self.current_step += 1
        rewards = [0.0] * self.num_agents

        # Execute actions
        new_positions = []
        for i, action in enumerate(actions):
            pos = self.agent_positions[i].copy()

            if action == 1 and pos[1] < self.grid_size - 1:  # up
                pos[1] += 1
            elif action == 2 and pos[1] > 0:  # down
                pos[1] -= 1
            elif action == 3 and pos[0] > 0:  # left
                pos[0] -= 1
            elif action == 4 and pos[0] < self.grid_size - 1:  # right
                pos[0] += 1
            # action == 0: stay

            new_positions.append(pos)

        # Check for collisions (agents can't occupy same cell)
        collision_agents = set()
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if new_positions[i] == new_positions[j]:
                    collision_agents.add(i)
                    collision_agents.add(j)

        # Apply movements (collision agents stay in place)
        for i in range(self.num_agents):
            if i not in collision_agents:
                self.agent_positions[i] = new_positions[i]
            else:
                rewards[i] -= 0.5  # Collision penalty

        # Check target collection
        targets_collected_this_step = []
        for i in range(self.num_agents):
            for j, target_pos in enumerate(self.target_positions):
                if (
                    not self.targets_collected[j]
                    and self.agent_positions[i] == target_pos
                ):
                    self.targets_collected[j] = True
                    rewards[i] += 10.0  # Target collection reward
                    targets_collected_this_step.append(j)

        # Team collaboration bonus
        if targets_collected_this_step:
            team_bonus = 2.0 * len(targets_collected_this_step)
            for i in range(self.num_agents):
                rewards[i] += team_bonus / self.num_agents

        # Small step penalty to encourage efficiency
        for i in range(self.num_agents):
            rewards[i] -= 0.1

        # Check termination
        done = (
            all(self.targets_collected) or self.current_step >= self.max_episode_steps
        )

        observations = self.get_observations()
        info = {
            "targets_collected": sum(self.targets_collected),
            "total_targets": self.num_targets,
            "collisions": len(collision_agents) // 2,
        }

        return observations, rewards, done, info
