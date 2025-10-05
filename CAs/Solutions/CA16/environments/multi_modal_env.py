"""
Multi-Modal Environment for Advanced RL

This module implements multi-modal environments that combine different types of observations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict as DictSpace


class MultiModalEnvironment(Env):
    """Environment that provides multi-modal observations."""

    def __init__(
        self,
        size: int = 8,
        num_goals: int = 3,
        num_obstacles: int = 5,
        include_visual: bool = True,
        include_audio: bool = True,
        include_text: bool = True,
    ):
        super().__init__()
        self.size = size
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.include_visual = include_visual
        self.include_audio = include_audio
        self.include_text = include_text

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = Discrete(4)

        # Multi-modal observation space
        self.observation_space = self._create_observation_space()

        # Environment state
        self.agent_pos = None
        self.goals = None
        self.obstacles = None
        self.grid = None

        # Multi-modal data
        self.visual_data = None
        self.audio_data = None
        self.text_data = None

        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 200

    def _create_observation_space(self) -> DictSpace:
        """Create multi-modal observation space."""
        spaces = {}
        
        # Basic state information
        spaces["state"] = Box(
            low=0, high=1, shape=(self.size * self.size + 2 + self.num_goals * 2,), dtype=np.float32
        )
        
        # Visual information
        if self.include_visual:
            spaces["visual"] = Box(
                low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8
            )
        
        # Audio information
        if self.include_audio:
            spaces["audio"] = Box(
                low=0, high=1, shape=(64,), dtype=np.float32
            )
        
        # Text information
        if self.include_text:
            spaces["text"] = Box(
                low=0, high=1, shape=(32,), dtype=np.float32
            )
        
        return DictSpace(spaces)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
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
            obstacle = [
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            ]
            if (
                obstacle != self.agent_pos
                and obstacle not in self.goals
                and obstacle not in self.obstacles
            ):
                self.obstacles.append(obstacle)

        # Initialize grid
        self.grid = np.zeros((self.size, self.size))
        for goal in self.goals:
            self.grid[goal[0], goal[1]] = 2  # Goal
        for obstacle in self.obstacles:
            self.grid[obstacle[0], obstacle[1]] = 1  # Obstacle

        # Generate multi-modal data
        self._generate_multimodal_data()

        # Reset episode tracking
        self.episode_length = 0

        return self.get_observation(), {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Actions: 0=up, 1=down, 2=left, 3=right
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
            else:
                reward = -0.1

            done = len(self.goals) == 0

        # Update episode length
        self.episode_length += 1

        # Check for episode termination
        if self.episode_length >= self.max_episode_length:
            done = True

        # Update multi-modal data
        self._update_multimodal_data()

        return self.get_observation(), reward, done, False, {}

    def _generate_multimodal_data(self):
        """Generate multi-modal data for current state."""
        # Visual data (RGB image)
        if self.include_visual:
            self.visual_data = self._generate_visual_data()
        
        # Audio data (spectrogram-like)
        if self.include_audio:
            self.audio_data = self._generate_audio_data()
        
        # Text data (embedding-like)
        if self.include_text:
            self.text_data = self._generate_text_data()

    def _generate_visual_data(self) -> np.ndarray:
        """Generate visual data (RGB image)."""
        # Create RGB array
        rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Set background color
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

    def _generate_audio_data(self) -> np.ndarray:
        """Generate audio data (spectrogram-like)."""
        # Simulate audio features based on environment state
        audio_features = np.zeros(64)
        
        # Distance to nearest goal
        if self.goals:
            distances = [
                abs(self.agent_pos[0] - goal[0]) + abs(self.agent_pos[1] - goal[1])
                for goal in self.goals
            ]
            min_distance = min(distances)
            audio_features[:16] = np.exp(-min_distance / 5.0)  # Closer = louder
        
        # Distance to nearest obstacle
        if self.obstacles:
            distances = [
                abs(self.agent_pos[0] - obs[0]) + abs(self.agent_pos[1] - obs[1])
                for obs in self.obstacles
            ]
            min_distance = min(distances)
            audio_features[16:32] = np.exp(-min_distance / 3.0)  # Closer = louder
        
        # Episode progress
        progress = self.episode_length / self.max_episode_length
        audio_features[32:48] = progress
        
        # Random noise
        audio_features[48:64] = np.random.normal(0, 0.1, 16)
        
        return audio_features.astype(np.float32)

    def _generate_text_data(self) -> np.ndarray:
        """Generate text data (embedding-like)."""
        # Simulate text embeddings based on environment state
        text_features = np.zeros(32)
        
        # Goal-related features
        if self.goals:
            text_features[:8] = 1.0  # Goals present
            text_features[8:16] = len(self.goals) / self.num_goals  # Goal count
        else:
            text_features[:8] = 0.0  # No goals
        
        # Obstacle-related features
        if self.obstacles:
            text_features[16:24] = 1.0  # Obstacles present
            text_features[24:32] = len(self.obstacles) / self.num_obstacles  # Obstacle count
        else:
            text_features[16:24] = 0.0  # No obstacles
        
        return text_features.astype(np.float32)

    def _update_multimodal_data(self):
        """Update multi-modal data after state change."""
        if self.include_visual:
            self.visual_data = self._generate_visual_data()
        if self.include_audio:
            self.audio_data = self._generate_audio_data()
        if self.include_text:
            self.text_data = self._generate_text_data()

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get multi-modal observation."""
        observation = {}
        
        # Basic state information
        observation["state"] = self._get_state_observation()
        
        # Multi-modal information
        if self.include_visual:
            observation["visual"] = self.visual_data
        if self.include_audio:
            observation["audio"] = self.audio_data
        if self.include_text:
            observation["text"] = self.text_data
        
        return observation

    def _get_state_observation(self) -> np.ndarray:
        """Get basic state observation."""
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
        
        # Combine all observations
        observation = np.concatenate([grid_flat, agent_pos, goal_positions])
        
        return observation.astype(np.float32)

    def get_multimodal_statistics(self) -> Dict[str, Any]:
        """Get multi-modal environment statistics."""
        stats = {
            "size": self.size,
            "num_goals": self.num_goals,
            "num_obstacles": self.num_obstacles,
            "include_visual": self.include_visual,
            "include_audio": self.include_audio,
            "include_text": self.include_text,
            "observation_space_size": len(self.observation_space.spaces),
        }
        
        # Add data statistics
        if self.include_visual and self.visual_data is not None:
            stats["visual_data_shape"] = self.visual_data.shape
            stats["visual_data_range"] = [self.visual_data.min(), self.visual_data.max()]
        
        if self.include_audio and self.audio_data is not None:
            stats["audio_data_shape"] = self.audio_data.shape
            stats["audio_data_range"] = [self.audio_data.min(), self.audio_data.max()]
        
        if self.include_text and self.text_data is not None:
            stats["text_data_shape"] = self.text_data.shape
            stats["text_data_range"] = [self.text_data.min(), self.text_data.max()]
        
        return stats

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            # Create display grid
            display_grid = np.zeros((self.size, self.size), dtype=str)
            display_grid.fill(".")

            # Place obstacles
            for obstacle in self.obstacles:
                display_grid[obstacle[0], obstacle[1]] = "X"

            # Place goals
            for goal in self.goals:
                display_grid[goal[0], goal[1]] = "G"

            # Place agent
            display_grid[self.agent_pos[0], self.agent_pos[1]] = "A"

            # Print grid
            print("\n" + "=" * (self.size * 2 + 1))
            for row in display_grid:
                print("|" + " ".join(row) + "|")
            print("=" * (self.size * 2 + 1))
            print(f"Agent: {self.agent_pos}, Goals: {self.goals}, Episode: {self.episode_length}")
            
            # Print multi-modal info
            if self.include_visual:
                print(f"Visual data shape: {self.visual_data.shape}")
            if self.include_audio:
                print(f"Audio data shape: {self.audio_data.shape}")
            if self.include_text:
                print(f"Text data shape: {self.text_data.shape}")

        elif mode == "rgb_array":
            # Return visual data if available
            if self.include_visual:
                return self.visual_data
            else:
                # Create simple RGB array
                rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
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