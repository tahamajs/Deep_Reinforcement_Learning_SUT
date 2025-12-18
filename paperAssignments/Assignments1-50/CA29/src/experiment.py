"""Experiment runner for SAC training and evaluation."""

import logging
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from .config import SACConfig
from .sac import SAC
from .utils import get_device, set_env_seed, set_seed


class Experiment:
    """Handles SAC training and evaluation experiments."""

    def __init__(self, config: SACConfig):
        """Initialize the experiment.

        Args:
            config: Configuration for the experiment.
        """
        self.config = config
        self.device = get_device(config.device)
        set_seed(config.seed)

        self.env = gym.make(config.env_name)
        set_env_seed(self.env, config.seed)

        self.test_env = gym.make(config.env_name)
        set_env_seed(self.test_env, config.seed + 1000)  # Different seed for evaluation

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.agent = SAC(state_dim, action_dim, config, self.device)

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def train(self) -> None:
        """Run the training loop."""
        self.logger.info("Starting training...")
        episode_reward = 0
        episode_num = 0
        state = self.env.reset()

        for step in range(self.config.num_steps):
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.agent.buffer.add(state, action, reward, next_state, done)
            self.agent.update()

            episode_reward += reward
            state = next_state

            if done:
                self.logger.info(f"Episode {episode_num}: Total Reward {episode_reward:.2f}")
                episode_reward = 0
                episode_num += 1
                state = self.env.reset()

            if (step + 1) % self.config.eval_freq == 0:
                self.evaluate(step + 1)

        self.logger.info("Training completed.")
        self.agent.save(str(self.log_dir / "sac_final.pth"))

    def evaluate(self, step: int, num_episodes: int = 10) -> float:
        """Evaluate the agent on the test environment.

        Args:
            step: Current training step (for logging).
            num_episodes: Number of evaluation episodes.

        Returns:
            Average reward over evaluation episodes.
        """
        rewards = []
        for _ in range(num_episodes):
            state = self.test_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state, deterministic=True)
                state, reward, done, _ = self.test_env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        self.logger.info(f"Step {step}: Eval Avg Reward {avg_reward:.2f} ± {std_reward:.2f}")
        return avg_reward

    def run_experiment(self) -> None:
        """Convenience method to run the full experiment."""
        self.train()