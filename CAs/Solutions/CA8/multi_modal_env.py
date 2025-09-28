"""
Multi-Modal Grid World Environment
Environment that combines visual, textual, and state information
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io


class PromptTemplate:
    """
    Template for generating textual prompts in multi-modal environments
    """

    def __init__(self):
        self.templates = {
            'navigation': "Navigate to the {color} {shape} in the {direction} corner.",
            'collection': "Collect the {color} {shape} while avoiding {obstacle}.",
            'puzzle': "Solve the puzzle by arranging {objects} in {pattern}."
        }

    def generate_prompt(self, task_type: str, **kwargs) -> str:
        """Generate a textual prompt for the given task"""
        if task_type not in self.templates:
            return "Complete the task successfully."

        template = self.templates[task_type]
        return template.format(**kwargs)

    def tokenize_prompt(self, prompt: str) -> Dict[str, Any]:
        """Tokenize prompt (simplified implementation)"""
        # Simplified tokenization - in practice would use proper tokenizer
        tokens = prompt.lower().split()
        token_ids = [hash(token) % 1000 for token in tokens]  # Simplified

        return {
            'tokens': token_ids,
            'mask': [1] * len(token_ids),
            'text': prompt
        }


class MultiModalGridWorld:
    """
    Multi-modal grid world environment with visual, textual, and state modalities
    """

    def __init__(self, size: int = 6, render_size: int = 84, max_steps: int = 100):
        self.size = size
        self.render_size = render_size
        self.max_steps = max_steps

        # Initialize prompt template
        self.prompt_template = PromptTemplate()

        # Reset environment
        self.reset()

        # Action space: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # Observation space: Multi-modal (visual + text + state)
        self.observation_space = spaces.Dict({
            'visual': spaces.Box(low=0, high=255, shape=(render_size, render_size, 3), dtype=np.uint8),
            'text': spaces.Dict({
                'tokens': spaces.MultiDiscrete([1000] * 20),  # Max 20 tokens
                'mask': spaces.MultiBinary(20),
                'text': spaces.Text(max_length=200)
            }),
            'state': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        })

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment"""
        # Random agent position
        self.agent_pos = np.random.randint(0, self.size, size=2)

        # Random goal position
        self.goal_pos = np.random.randint(0, self.size, size=2)
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.random.randint(0, self.size, size=2)

        # Random obstacles
        self.obstacles = []
        for _ in range(self.size // 2):
            obstacle = np.random.randint(0, self.size, size=2)
            while (np.array_equal(obstacle, self.agent_pos) or
                   np.array_equal(obstacle, self.goal_pos) or
                   any(np.array_equal(obstacle, obs) for obs in self.obstacles)):
                obstacle = np.random.randint(0, self.size, size=2)
            self.obstacles.append(obstacle)

        self.steps = 0
        self.task_type = np.random.choice(['navigation', 'collection', 'puzzle'])

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        action_map = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1])    # Right
        }

        # Update position
        new_pos = self.agent_pos + action_map[action]

        # Check bounds
        if (new_pos >= 0).all() and (new_pos < self.size).all():
            # Check obstacles
            if not any(np.array_equal(new_pos, obs) for obs in self.obstacles):
                self.agent_pos = new_pos

        self.steps += 1

        # Compute reward
        reward = -0.01  # Step penalty

        # Goal reached
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 1.0

        # Obstacle collision
        if any(np.array_equal(self.agent_pos, obs) for obs in self.obstacles):
            reward -= 0.5

        # Termination conditions
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.steps >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self) -> Dict[str, Any]:
        """Get multi-modal observation"""
        # Visual observation
        visual_obs = self._render_visual()

        # Textual observation
        text_obs = self._generate_instruction()

        # State observation
        state_obs = self.agent_pos.copy()

        return {
            'visual': visual_obs,
            'text': text_obs,
            'state': state_obs
        }

    def _render_visual(self) -> np.ndarray:
        """Render visual observation"""
        # Create RGB image
        img = Image.new('RGB', (self.render_size, self.render_size), color='white')
        draw = ImageDraw.Draw(img)

        # Cell size
        cell_size = self.render_size // self.size

        # Draw grid
        for i in range(self.size + 1):
            # Horizontal lines
            draw.line([0, i * cell_size, self.render_size, i * cell_size],
                     fill='black', width=1)
            # Vertical lines
            draw.line([i * cell_size, 0, i * cell_size, self.render_size],
                     fill='black', width=1)

        # Draw obstacles
        for obs in self.obstacles:
            x, y = obs[1] * cell_size, obs[0] * cell_size
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill='red')

        # Draw goal
        x, y = self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size
        draw.rectangle([x, y, x + cell_size, y + cell_size], fill='green')

        # Draw agent
        x, y = self.agent_pos[1] * cell_size, self.agent_pos[0] * cell_size
        draw.rectangle([x, y, x + cell_size, y + cell_size], fill='blue')

        # Convert to numpy array
        img_array = np.array(img)

        return img_array

    def _generate_instruction(self) -> Dict[str, Any]:
        """Generate textual instruction"""
        if self.task_type == 'navigation':
            direction = self._get_direction_to_goal()
            instruction = self.prompt_template.generate_prompt(
                'navigation',
                color='green',
                shape='square',
                direction=direction
            )
        elif self.task_type == 'collection':
            instruction = self.prompt_template.generate_prompt(
                'collection',
                color='green',
                shape='square',
                obstacle='red squares'
            )
        else:  # puzzle
            instruction = self.prompt_template.generate_prompt(
                'puzzle',
                objects='colored squares',
                pattern='specific arrangement'
            )

        return self.prompt_template.tokenize_prompt(instruction)

    def _get_direction_to_goal(self) -> str:
        """Get directional description to goal"""
        dx = self.goal_pos[0] - self.agent_pos[0]
        dy = self.goal_pos[1] - self.agent_pos[1]

        if abs(dx) > abs(dy):
            if dx > 0:
                return "bottom"
            else:
                return "top"
        else:
            if dy > 0:
                return "right"
            else:
                return "left"

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == 'rgb_array':
            return self._render_visual()
        elif mode == 'human':
            plt.imshow(self._render_visual())
            plt.axis('off')
            plt.show()
        return None

    def close(self):
        """Close the environment"""
        pass


class MultiModalWrapper:
    """
    Wrapper that processes multi-modal observations for RL agents
    """

    def __init__(self, env: MultiModalGridWorld):
        self.env = env

        # Feature dimensions
        self.visual_dim = 64  # After CNN processing
        self.text_dim = 32    # After transformer processing
        self.state_dim = 2    # Raw state

        self.total_dim = self.visual_dim + self.text_dim + self.state_dim

    def process_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Process multi-modal observation into feature vector

        Args:
            obs: Multi-modal observation

        Returns:
            Processed feature vector
        """
        # Process visual (simplified - would use CNN in practice)
        visual_features = self._process_visual(obs['visual'])

        # Process text (simplified - would use transformer in practice)
        text_features = self._process_text(obs['text'])

        # State features (normalized)
        state_features = obs['state'].astype(np.float32) / self.env.size

        # Concatenate
        features = np.concatenate([visual_features, text_features, state_features])

        return features

    def _process_visual(self, visual_obs: np.ndarray) -> np.ndarray:
        """Process visual observation (simplified)"""
        # Simple downsampling and flattening
        resized = visual_obs[::4, ::4, :]  # Downsample
        features = resized.mean(axis=(0, 1))  # Average pooling
        return features / 255.0  # Normalize

    def _process_text(self, text_obs: Dict[str, Any]) -> np.ndarray:
        """Process textual observation (simplified)"""
        # Simple averaging of token embeddings
        tokens = np.array(text_obs['tokens'])
        mask = np.array(text_obs['mask'])

        # Masked average
        masked_tokens = tokens * mask
        features = masked_tokens.sum(axis=0, keepdims=True) / mask.sum()
        features = np.tile(features, (32, 1))[:32].flatten()  # Pad/truncate to 32 dims

        return features / 1000.0  # Normalize