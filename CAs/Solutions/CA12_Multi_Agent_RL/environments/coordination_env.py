"""
Coordination environment implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional


class CoordinationEnvironment:
    """Environment requiring coordination between agents."""

    def __init__(self, n_agents: int = 3, coordination_threshold: float = 0.7):
        self.n_agents = n_agents
        self.coordination_threshold = coordination_threshold
        self.state_dim = n_agents * 2
        self.action_dim = 2  # Binary actions

        self.state = None
        self.current_step = 0
        self.max_steps = 50

    def reset(self) -> List[np.ndarray]:
        """Reset environment."""
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.current_step = 0

        observations = [self.state.copy() for _ in range(self.n_agents)]
        return observations

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        """Execute actions."""
        # Compute coordination level
        action_consensus = np.mean(actions)
        coordination_level = 1.0 - abs(action_consensus - 0.5) * 2

        # State evolution based on coordination
        coordination_bonus = coordination_level * 0.1
        self.state += np.random.normal(0, 0.05, self.state_dim) + coordination_bonus

        # Rewards based on coordination
        if coordination_level > self.coordination_threshold:
            rewards = [1.0] * self.n_agents  # High reward for good coordination
        else:
            rewards = [-0.1] * self.n_agents  # Penalty for poor coordination

        self.current_step += 1
        done = self.current_step >= self.max_steps

        observations = [self.state.copy() for _ in range(self.n_agents)]
        info = {
            "coordination_level": coordination_level,
            "action_consensus": action_consensus,
        }

        return observations, rewards, done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render coordination environment."""
        if mode == "human":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot state evolution
            ax1.plot(self.state)
            ax1.set_title("State Evolution")
            ax1.set_xlabel("State Dimension")
            ax1.set_ylabel("State Value")
            ax1.grid(True)

            # Plot coordination metric
            coordination_history = getattr(self, "coordination_history", [])
            if coordination_history:
                ax2.plot(coordination_history)
                ax2.axhline(
                    y=self.coordination_threshold,
                    color="r",
                    linestyle="--",
                    label=f"Threshold ({self.coordination_threshold})",
                )
                ax2.set_title("Coordination Level")
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Coordination")
                ax2.legend()
                ax2.grid(True)

            plt.tight_layout()
            plt.show()

        return None
