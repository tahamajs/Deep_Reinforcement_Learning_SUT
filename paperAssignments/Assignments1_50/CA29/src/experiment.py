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

        # Simple CSV metrics file to store evaluation summaries (step, avg, std)
        self.metrics_file = self.log_dir / 'metrics.csv'
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write('step,avg_reward,std_reward\n')

    def train(self) -> None:
        """Run the training loop.

        This method is robust to both Gym and Gymnasium APIs. Gymnasium's
        `reset` can return (obs, info) and `step` can return
        (obs, reward, terminated, truncated, info). We handle both formats.
        """
        self.logger.info("Starting training...")
        episode_reward = 0
        episode_num = 0
        reset_out = self.env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        for step in range(self.config.num_steps):
            action = self.agent.select_action(state)
            step_out = self.env.step(action)
            # Support both Gym and Gymnasium step signatures
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, _ = step_out

            self.agent.buffer.add(state, action, float(reward), next_state, bool(done))
            self.agent.update()

            episode_reward += float(reward)
            state = next_state

            if done:
                self.logger.info(f"Episode {episode_num}: Total Reward {episode_reward:.2f}")
                episode_reward = 0
                episode_num += 1
                reset_out = self.env.reset()
                state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

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
            reset_out = self.test_env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            episode_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state, deterministic=True)
                step_out = self.test_env.step(action)
                if len(step_out) == 5:
                    state, reward, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    state, reward, done, _ = step_out
                episode_reward += float(reward)
            rewards.append(episode_reward)

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        self.logger.info(f"Step {step}: Eval Avg Reward {avg_reward:.2f} ± {std_reward:.2f}")
        # Append to CSV for easy plotting
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(f"{step},{avg_reward:.6f},{std_reward:.6f}\n")
        except Exception:
            self.logger.exception("Failed to write metrics file")
        return avg_reward

    def run_experiment(self) -> None:
        """Convenience method to run the full experiment."""
        self.train()