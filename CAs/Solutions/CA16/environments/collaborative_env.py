"""
Collaborative Environment for Human-AI Collaboration

This module implements a collaborative grid world environment for testing human-AI collaboration.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


class CollaborativeGridWorld(Env):
    """Collaborative grid world environment for human-AI collaboration."""
    
    def __init__(self, size: int = 8, num_goals: int = 3, num_obstacles: int = 5):
        super().__init__()
        self.size = size
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = Discrete(4)
        
        # Observation space: flattened grid + agent position + goal positions + collaboration info
        self.observation_space = Box(
            low=0, high=1, shape=(size * size + 2 + num_goals * 2 + 3,), dtype=np.float32
        )
        
        # Environment state
        self.agent_pos = None
        self.goals = None
        self.obstacles = None
        self.grid = None
        
        # Collaboration state
        self.human_available = True
        self.collaboration_history = []
        self.human_confidence = 0.8
        self.ai_confidence = 0.7
        
        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 200
        
        # Collaboration metrics
        self.collaboration_metrics = {
            'human_interventions': 0,
            'ai_decisions': 0,
            'collaborative_decisions': 0,
            'successful_collaborations': 0
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize agent position
        self.agent_pos = [0, 0]
        
        # Initialize goals
        self.goals = []
        while len(self.goals) < self.num_goals:
            goal = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            if goal != self.agent_pos and goal not in self.goals:
                self.goals.append(goal)
        
        # Initialize obstacles
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            obstacle = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            if (obstacle != self.agent_pos and obstacle not in self.goals and 
                obstacle not in self.obstacles):
                self.obstacles.append(obstacle)
        
        # Initialize grid
        self.grid = np.zeros((self.size, self.size))
        for goal in self.goals:
            self.grid[goal[0], goal[1]] = 2  # Goal
        for obstacle in self.obstacles:
            self.grid[obstacle[0], obstacle[1]] = 1  # Obstacle
        
        # Reset collaboration state
        self.collaboration_history = []
        self.collaboration_metrics = {
            'human_interventions': 0,
            'ai_decisions': 0,
            'collaborative_decisions': 0,
            'successful_collaborations': 0
        }
        
        # Reset episode tracking
        self.episode_length = 0
        
        return self.get_observation(), {}
    
    def step(self, action: int, human_action: Optional[int] = None, 
             human_confidence: float = 0.5) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Record collaboration
        if human_action is not None:
            self.collaboration_history.append({
                'ai_action': action,
                'human_action': human_action,
                'human_confidence': human_confidence,
                'used_human': np.random.random() < human_confidence
            })
            
            # Use human action if confident enough
            if self.collaboration_history[-1]['used_human']:
                action = human_action
                self.collaboration_metrics['human_interventions'] += 1
            else:
                self.collaboration_metrics['ai_decisions'] += 1
        else:
            self.collaboration_metrics['ai_decisions'] += 1
        
        # Execute action
        new_pos = self.agent_pos.copy()
        
        if action == 0:  # up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 2:  # left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        
        # Check for obstacles
        if new_pos in self.obstacles:
            reward = -1
            done = False
        else:
            self.agent_pos = new_pos
            
            # Check for goals
            if new_pos in self.goals:
                self.goals.remove(new_pos)
                reward = 10
                
                # Bonus for successful collaboration
                if human_action is not None and self.collaboration_history[-1]['used_human']:
                    reward += 1.0
                    self.collaboration_metrics['successful_collaborations'] += 1
            else:
                reward = -0.1
            
            done = len(self.goals) == 0
        
        # Update episode length
        self.episode_length += 1
        
        # Check for episode termination
        if self.episode_length >= self.max_episode_length:
            done = True
        
        # Create info dictionary
        info = {
            'collaboration_used': human_action is not None,
            'collaboration_history': self.collaboration_history.copy(),
            'collaboration_metrics': self.collaboration_metrics.copy()
        }
        
        return self.get_observation(), reward, done, False, info
    
    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Flatten grid
        grid_flat = self.grid.flatten()
        
        # Agent position
        agent_pos = np.array(self.agent_pos, dtype=np.float32) / self.size
        
        # Goal positions
        goal_positions = np.zeros(self.num_goals * 2, dtype=np.float32)
        for i, goal in enumerate(self.goals):
            if i < self.num_goals:
                goal_positions[i * 2] = goal[0] / self.size
                goal_positions[i * 2 + 1] = goal[1] / self.size
        
        # Collaboration information
        collaboration_info = np.array([
            float(self.human_available),
            self.human_confidence,
            self.ai_confidence
        ], dtype=np.float32)
        
        # Combine all observations
        observation = np.concatenate([grid_flat, agent_pos, goal_positions, collaboration_info])
        
        return observation.astype(np.float32)
    
    def set_human_availability(self, available: bool):
        """Set human availability."""
        self.human_available = available
    
    def set_human_confidence(self, confidence: float):
        """Set human confidence level."""
        self.human_confidence = max(0.0, min(1.0, confidence))
    
    def set_ai_confidence(self, confidence: float):
        """Set AI confidence level."""
        self.ai_confidence = max(0.0, min(1.0, confidence))
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        total_decisions = (self.collaboration_metrics['human_interventions'] + 
                          self.collaboration_metrics['ai_decisions'])
        
        if total_decisions == 0:
            return self.collaboration_metrics.copy()
        
        stats = self.collaboration_metrics.copy()
        stats['human_intervention_rate'] = self.collaboration_metrics['human_interventions'] / total_decisions
        stats['ai_decision_rate'] = self.collaboration_metrics['ai_decisions'] / total_decisions
        stats['collaboration_success_rate'] = (
            self.collaboration_metrics['successful_collaborations'] / 
            max(1, self.collaboration_metrics['human_interventions'])
        )
        
        return stats
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'human':
            # Create display grid
            display_grid = np.zeros((self.size, self.size), dtype=str)
            display_grid.fill('.')
            
            # Place obstacles
            for obstacle in self.obstacles:
                display_grid[obstacle[0], obstacle[1]] = 'X'
            
            # Place goals
            for goal in self.goals:
                display_grid[goal[0], goal[1]] = 'G'
            
            # Place agent
            display_grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
            
            # Print grid
            print("\n" + "=" * (self.size * 2 + 1))
            for row in display_grid:
                print("|" + " ".join(row) + "|")
            print("=" * (self.size * 2 + 1))
            print(f"Agent: {self.agent_pos}, Goals: {self.goals}, Episode: {self.episode_length}")
            print(f"Human Available: {self.human_available}, Human Confidence: {self.human_confidence:.2f}")
            print(f"AI Confidence: {self.ai_confidence:.2f}")
            
            # Print collaboration statistics
            stats = self.get_collaboration_statistics()
            print(f"Collaboration Stats: {stats}")
            
        elif mode == 'rgb_array':
            # Create RGB array
            rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            
            # Set colors
            rgb_array[:, :] = [255, 255, 255]  # White background
            
            # Obstacles (black)
            for obstacle in self.obstacles:
                rgb_array[obstacle[0], obstacle[1]] = [0, 0, 0]
            
            # Goals (green)
            for goal in self.goals:
                rgb_array[goal[0], goal[1]] = [0, 255, 0]
            
            # Agent (red)
            rgb_array[self.agent_pos[0], self.agent_pos[1]] = [255, 0, 0]
            
            return rgb_array
        
        return None
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            'size': self.size,
            'num_goals': self.num_goals,
            'num_obstacles': self.num_obstacles,
            'agent_pos': self.agent_pos,
            'goals': self.goals,
            'obstacles': self.obstacles,
            'episode_length': self.episode_length,
            'human_available': self.human_available,
            'human_confidence': self.human_confidence,
            'ai_confidence': self.ai_confidence,
            'collaboration_history': self.collaboration_history.copy(),
            'collaboration_metrics': self.collaboration_metrics.copy()
        }
    
    def simulate_human_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Simulate human action selection."""
        # Simple heuristic: move towards nearest goal
        if not self.goals:
            return np.random.randint(0, 4), 0.5
        
        # Find nearest goal
        nearest_goal = min(self.goals, key=lambda g: abs(g[0] - self.agent_pos[0]) + abs(g[1] - self.agent_pos[1]))
        
        # Determine best action
        if nearest_goal[0] > self.agent_pos[0]:  # Goal is below
            action = 1  # down
        elif nearest_goal[0] < self.agent_pos[0]:  # Goal is above
            action = 0  # up
        elif nearest_goal[1] > self.agent_pos[1]:  # Goal is to the right
            action = 3  # right
        elif nearest_goal[1] < self.agent_pos[1]:  # Goal is to the left
            action = 2  # left
        else:
            action = np.random.randint(0, 4)
        
        # Simulate confidence (higher for closer goals)
        distance = abs(nearest_goal[0] - self.agent_pos[0]) + abs(nearest_goal[1] - self.agent_pos[1])
        confidence = max(0.3, 1.0 - distance / (self.size * 2))
        
        return action, confidence
    
    def get_collaboration_reward(self, ai_action: int, human_action: int, 
                                human_confidence: float) -> float:
        """Get reward for collaboration."""
        # Base reward
        reward = 0.0
        
        # Reward for human intervention
        if human_confidence > 0.7:
            reward += 0.1
        
        # Reward for agreement
        if ai_action == human_action:
            reward += 0.2
        
        # Penalty for disagreement
        if ai_action != human_action:
            reward -= 0.1
        
        return reward