"""
Complex Environments for Advanced DQN Methods
Includes: Multi-agent environments, dynamic environments, and complex reward structures
"""

import gym
import numpy as np
from gym import spaces
from typing import Tuple, Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import random


class MultiAgentGridWorld(gym.Env):
    """
    Multi-agent Grid World with competitive/cooperative dynamics
    """
    
    def __init__(self, size: int = 8, num_agents: int = 2, mode: str = "cooperative"):
        super().__init__()
        self.size = size
        self.num_agents = num_agents
        self.mode = mode  # "cooperative", "competitive", "mixed"
        
        # Action space: 4 directions + stay
        self.action_space = spaces.Discrete(5)
        # Observation space: flattened grid + agent positions
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size * size + num_agents * 2,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.agent_positions = []
        self.goals = []
        
        # Place agents randomly
        for i in range(self.num_agents):
            while True:
                pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                if pos not in self.agent_positions:
                    self.agent_positions.append(pos)
                    break
        
        # Place goals
        for i in range(self.num_agents):
            while True:
                goal = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                if goal not in self.goals and goal not in self.agent_positions:
                    self.goals.append(goal)
                    break
        
        # Place obstacles
        self.obstacles = []
        num_obstacles = random.randint(2, 5)
        for _ in range(num_obstacles):
            while True:
                obs = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                if obs not in self.agent_positions and obs not in self.goals:
                    self.obstacles.append(obs)
                    break
        
        # Update grid
        self._update_grid()
        
        return self._get_observation()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """Execute actions for all agents"""
        rewards = []
        dones = []
        infos = []
        
        # Move agents
        for i, action in enumerate(actions):
            if i < len(self.agent_positions):
                new_pos = self._move_agent(i, action)
                self.agent_positions[i] = new_pos
        
        # Update grid
        self._update_grid()
        
        # Calculate rewards
        for i in range(self.num_agents):
            reward, done, info = self._calculate_reward(i)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        observations = [self._get_observation() for _ in range(self.num_agents)]
        
        return observations, rewards, dones, infos
    
    def _move_agent(self, agent_id: int, action: int) -> List[int]:
        """Move agent based on action"""
        current_pos = self.agent_positions[agent_id]
        
        if action == 0:  # Up
            new_pos = [max(0, current_pos[0] - 1), current_pos[1]]
        elif action == 1:  # Down
            new_pos = [min(self.size - 1, current_pos[0] + 1), current_pos[1]]
        elif action == 2:  # Left
            new_pos = [current_pos[0], max(0, current_pos[1] - 1)]
        elif action == 3:  # Right
            new_pos = [current_pos[0], min(self.size - 1, current_pos[1] + 1)]
        else:  # Stay
            new_pos = current_pos
        
        # Check collision with obstacles
        if new_pos in self.obstacles:
            return current_pos
        
        # Check collision with other agents
        for i, other_pos in enumerate(self.agent_positions):
            if i != agent_id and new_pos == other_pos:
                return current_pos
        
        return new_pos
    
    def _calculate_reward(self, agent_id: int) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate reward for agent"""
        agent_pos = self.agent_positions[agent_id]
        goal_pos = self.goals[agent_id]
        
        # Distance to goal
        distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        
        # Base reward
        if agent_pos == goal_pos:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # Step penalty
            done = False
        
        # Multi-agent dynamics
        if self.mode == "cooperative":
            # Reward for helping other agents
            for i, other_pos in enumerate(self.agent_positions):
                if i != agent_id and other_pos == self.goals[i]:
                    reward += 2.0
        elif self.mode == "competitive":
            # Penalty for other agents reaching goals
            for i, other_pos in enumerate(self.agent_positions):
                if i != agent_id and other_pos == self.goals[i]:
                    reward -= 1.0
        
        info = {
            "agent_id": agent_id,
            "position": agent_pos,
            "goal": goal_pos,
            "distance_to_goal": distance
        }
        
        return reward, done, info
    
    def _update_grid(self):
        """Update grid representation"""
        self.grid.fill(0)
        
        # Place obstacles
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = 0.3
        
        # Place goals
        for i, goal in enumerate(self.goals):
            self.grid[goal[0], goal[1]] = 0.5 + i * 0.1
        
        # Place agents
        for i, pos in enumerate(self.agent_positions):
            self.grid[pos[0], pos[1]] = 0.8 + i * 0.05
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for agent"""
        # Flatten grid
        grid_flat = self.grid.flatten()
        
        # Add agent positions
        agent_info = []
        for pos in self.agent_positions:
            agent_info.extend([pos[0] / self.size, pos[1] / self.size])
        
        return np.concatenate([grid_flat, agent_info]).astype(np.float32)
    
    def render(self, mode: str = "human") -> Any:
        """Render environment"""
        if mode == "human":
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Draw grid
            for i in range(self.size + 1):
                ax.axhline(i - 0.5, color='black', linewidth=0.5)
                ax.axvline(i - 0.5, color='black', linewidth=0.5)
            
            # Draw obstacles
            for obs in self.obstacles:
                rect = patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                                       linewidth=1, edgecolor='black', facecolor='gray')
                ax.add_patch(rect)
            
            # Draw goals
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            for i, goal in enumerate(self.goals):
                circle = patches.Circle((goal[1], goal[0]), 0.3, color=colors[i % len(colors)])
                ax.add_patch(circle)
            
            # Draw agents
            for i, pos in enumerate(self.agent_positions):
                triangle = patches.RegularPolygon((pos[1], pos[0]), 3, radius=0.2,
                                                color=colors[i % len(colors)])
                ax.add_patch(triangle)
            
            ax.set_xlim(-0.5, self.size - 0.5)
            ax.set_ylim(-0.5, self.size - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f"Multi-Agent Grid World ({self.mode})")
            plt.show()


class DynamicEnvironment(gym.Env):
    """
    Dynamic environment with changing obstacles and goals
    """
    
    def __init__(self, size: int = 10, change_frequency: int = 50):
        super().__init__()
        self.size = size
        self.change_frequency = change_frequency
        self.step_count = 0
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size, size, 3), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.obstacles = []
        self.step_count = 0
        
        # Generate initial obstacles
        self._generate_obstacles()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action"""
        self.step_count += 1
        
        # Move agent
        self._move_agent(action)
        
        # Change environment periodically
        if self.step_count % self.change_frequency == 0:
            self._change_environment()
        
        # Calculate reward
        reward, done, info = self._calculate_reward()
        
        return self._get_observation(), reward, done, info
    
    def _move_agent(self, action: int):
        """Move agent"""
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        dx, dy = action_map[action]
        new_pos = [
            max(0, min(self.size - 1, self.agent_pos[0] + dx)),
            max(0, min(self.size - 1, self.agent_pos[1] + dy))
        ]
        
        # Check collision with obstacles
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
    
    def _change_environment(self):
        """Change environment dynamics"""
        # Move goal randomly
        while True:
            new_goal = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            if new_goal != self.agent_pos and new_goal not in self.obstacles:
                self.goal_pos = new_goal
                break
        
        # Regenerate obstacles
        self._generate_obstacles()
    
    def _generate_obstacles(self):
        """Generate random obstacles"""
        self.obstacles = []
        num_obstacles = random.randint(3, 8)
        
        for _ in range(num_obstacles):
            while True:
                obs = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
                if obs != self.agent_pos and obs != self.goal_pos and obs not in self.obstacles:
                    self.obstacles.append(obs)
                    break
    
    def _calculate_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate reward"""
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            # Distance-based reward
            distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1 - distance * 0.01
            done = False
        
        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "step_count": self.step_count,
            "obstacles": len(self.obstacles)
        }
        
        return reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation"""
        obs = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Agent position (red channel)
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1.0
        
        # Goal position (green channel)
        obs[self.goal_pos[0], self.goal_pos[1], 1] = 1.0
        
        # Obstacles (blue channel)
        for obs_pos in self.obstacles:
            obs[obs_pos[0], obs_pos[1], 2] = 1.0
        
        return obs


class HierarchicalEnvironment(gym.Env):
    """
    Hierarchical environment with sub-goals and meta-goals
    """
    
    def __init__(self, size: int = 12, num_subgoals: int = 3):
        super().__init__()
        self.size = size
        self.num_subgoals = num_subgoals
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size * size + num_subgoals * 2 + 2,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.agent_pos = [0, 0]
        self.meta_goal = [self.size - 1, self.size - 1]
        self.subgoals = []
        self.current_subgoal_idx = 0
        
        # Generate subgoals
        self._generate_subgoals()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action"""
        # Move agent
        self._move_agent(action)
        
        # Calculate reward
        reward, done, info = self._calculate_reward()
        
        return self._get_observation(), reward, done, info
    
    def _move_agent(self, action: int):
        """Move agent"""
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        dx, dy = action_map[action]
        new_pos = [
            max(0, min(self.size - 1, self.agent_pos[0] + dx)),
            max(0, min(self.size - 1, self.agent_pos[1] + dy))
        ]
        
        self.agent_pos = new_pos
    
    def _generate_subgoals(self):
        """Generate subgoals"""
        self.subgoals = []
        
        # Create path from start to meta-goal
        path_points = []
        current = self.agent_pos.copy()
        target = self.meta_goal.copy()
        
        # Generate intermediate points
        for i in range(self.num_subgoals):
            progress = (i + 1) / (self.num_subgoals + 1)
            point = [
                int(current[0] + progress * (target[0] - current[0])),
                int(current[1] + progress * (target[1] - current[1]))
            ]
            path_points.append(point)
        
        self.subgoals = path_points
    
    def _calculate_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate hierarchical reward"""
        reward = 0.0
        done = False
        
        # Check if reached current subgoal
        if self.current_subgoal_idx < len(self.subgoals):
            current_subgoal = self.subgoals[self.current_subgoal_idx]
            
            if self.agent_pos == current_subgoal:
                reward += 5.0  # Subgoal reward
                self.current_subgoal_idx += 1
                
                # Check if all subgoals reached
                if self.current_subgoal_idx >= len(self.subgoals):
                    # Check if reached meta-goal
                    if self.agent_pos == self.meta_goal:
                        reward += 20.0  # Meta-goal reward
                        done = True
                    else:
                        reward += 2.0  # All subgoals completed
        else:
            # All subgoals reached, check meta-goal
            if self.agent_pos == self.meta_goal:
                reward += 20.0
                done = True
        
        # Distance-based shaping
        if not done:
            if self.current_subgoal_idx < len(self.subgoals):
                target = self.subgoals[self.current_subgoal_idx]
            else:
                target = self.meta_goal
            
            distance = abs(self.agent_pos[0] - target[0]) + abs(self.agent_pos[1] - target[1])
            reward -= distance * 0.01
        
        info = {
            "agent_pos": self.agent_pos,
            "current_subgoal": self.subgoals[self.current_subgoal_idx] if self.current_subgoal_idx < len(self.subgoals) else None,
            "meta_goal": self.meta_goal,
            "subgoals_completed": self.current_subgoal_idx
        }
        
        return reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation"""
        # Grid representation
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1.0
        grid[self.meta_goal[0], self.meta_goal[1]] = 0.5
        
        for i, subgoal in enumerate(self.subgoals):
            if i >= self.current_subgoal_idx:
                grid[subgoal[0], subgoal[1]] = 0.3
        
        # Flatten grid
        grid_flat = grid.flatten()
        
        # Add agent position
        agent_info = [self.agent_pos[0] / self.size, self.agent_pos[1] / self.size]
        
        # Add subgoal information
        subgoal_info = []
        for subgoal in self.subgoals:
            subgoal_info.extend([subgoal[0] / self.size, subgoal[1] / self.size])
        
        return np.concatenate([grid_flat, agent_info, subgoal_info]).astype(np.float32)


class StochasticEnvironment(gym.Env):
    """
    Stochastic environment with probabilistic transitions
    """
    
    def __init__(self, size: int = 8, stochastic_prob: float = 0.3):
        super().__init__()
        self.size = size
        self.stochastic_prob = stochastic_prob
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size, size), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.state = np.zeros((self.size, self.size), dtype=np.float32)
        self.state[self.agent_pos[0], self.agent_pos[1]] = 1.0
        self.state[self.goal_pos[0], self.goal_pos[1]] = 0.5
        
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action with stochastic transitions"""
        # Stochastic action selection
        if random.random() < self.stochastic_prob:
            # Random action instead of intended action
            action = random.randint(0, 3)
        
        # Move agent
        self._move_agent(action)
        
        # Calculate reward
        reward, done, info = self._calculate_reward()
        
        return self.state, reward, done, info
    
    def _move_agent(self, action: int):
        """Move agent"""
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        dx, dy = action_map[action]
        new_pos = [
            max(0, min(self.size - 1, self.agent_pos[0] + dx)),
            max(0, min(self.size - 1, self.agent_pos[1] + dy))
        ]
        
        # Update state
        self.state[self.agent_pos[0], self.agent_pos[1]] = 0.0
        self.agent_pos = new_pos
        self.state[self.agent_pos[0], self.agent_pos[1]] = 1.0
    
    def _calculate_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculate reward"""
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            reward = -0.1
            done = False
        
        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "stochastic_prob": self.stochastic_prob
        }
        
        return reward, done, info


def make_complex_env(env_name: str, **kwargs) -> gym.Env:
    """Factory function for complex environments"""
    if env_name == "MultiAgentGridWorld":
        return MultiAgentGridWorld(**kwargs)
    elif env_name == "DynamicEnvironment":
        return DynamicEnvironment(**kwargs)
    elif env_name == "HierarchicalEnvironment":
        return HierarchicalEnvironment(**kwargs)
    elif env_name == "StochasticEnvironment":
        return StochasticEnvironment(**kwargs)
    else:
        raise ValueError(f"Unknown complex environment: {env_name}")


if __name__ == "__main__":
    print("Complex environments loaded successfully!")
    print("Available environments:")
    print("- MultiAgentGridWorld: Multi-agent scenarios")
    print("- DynamicEnvironment: Changing obstacles and goals")
    print("- HierarchicalEnvironment: Sub-goals and meta-goals")
    print("- StochasticEnvironment: Probabilistic transitions")
