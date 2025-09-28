"""
Exploration strategies for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch


class ExplorationStrategy:
    """Base class for exploration strategies"""

    def __init__(self):
        """Initialize exploration strategy"""
        pass

    def get_action(self, action_probs: np.ndarray) -> int:
        """Select action based on exploration strategy

        Args:
            action_probs: Action probabilities

        Returns:
            Selected action
        """
        raise NotImplementedError

    def update(self, **kwargs):
        """Update exploration parameters"""
        pass


class EpsilonGreedyExploration(ExplorationStrategy):
    """ε-greedy exploration for policy gradients"""

    def __init__(
        self, epsilon: float = 0.1, decay_rate: float = 0.995, min_epsilon: float = 0.01
    ):
        """Initialize ε-greedy exploration

        Args:
            epsilon: Initial exploration rate
            decay_rate: Rate at which epsilon decays
            min_epsilon: Minimum epsilon value
        """
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.episode_count = 0

    def get_action(self, action_probs: np.ndarray) -> int:
        """Select action with ε-greedy strategy

        Args:
            action_probs: Action probabilities from policy

        Returns:
            Selected action
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(action_probs))
        else:
            return np.argmax(action_probs)

    def update(self, episode_done: bool = False):
        """Update exploration rate

        Args:
            episode_done: Whether episode ended
        """
        if episode_done:
            self.episode_count += 1
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class BoltzmannExploration(ExplorationStrategy):
    """Boltzmann exploration (temperature-based)"""

    def __init__(
        self, temperature: float = 1.0, decay_rate: float = 0.995, min_temp: float = 0.1
    ):
        """Initialize Boltzmann exploration

        Args:
            temperature: Initial temperature
            decay_rate: Temperature decay rate
            min_temp: Minimum temperature
        """
        self.initial_temp = temperature
        self.temperature = temperature
        self.decay_rate = decay_rate
        self.min_temp = min_temp
        self.episode_count = 0

    def get_action(self, action_probs: np.ndarray) -> int:
        """Select action using Boltzmann distribution

        Args:
            action_probs: Action probabilities from policy

        Returns:
            Selected action
        """
        # Apply temperature scaling
        scaled_probs = action_probs ** (1.0 / self.temperature)
        scaled_probs = scaled_probs / np.sum(scaled_probs)

        return np.random.choice(len(action_probs), p=scaled_probs)

    def update(self, episode_done: bool = False):
        """Update temperature

        Args:
            episode_done: Whether episode ended
        """
        if episode_done:
            self.episode_count += 1
            self.temperature = max(self.min_temp, self.temperature * self.decay_rate)


class EntropyBonusExploration:
    """Entropy bonus for encouraging exploration"""

    def __init__(self, entropy_coeff: float = 0.01):
        """Initialize entropy bonus

        Args:
            entropy_coeff: Coefficient for entropy bonus
        """
        self.entropy_coeff = entropy_coeff

    def compute_entropy(self, action_probs: np.ndarray) -> float:
        """Compute entropy of action distribution

        Args:
            action_probs: Action probabilities

        Returns:
            Entropy value
        """
        # Avoid log(0) by adding small epsilon
        probs = np.clip(action_probs, 1e-8, 1.0)
        return -np.sum(probs * np.log(probs))

    def get_entropy_bonus(self, action_probs: np.ndarray) -> float:
        """Get entropy bonus for reward

        Args:
            action_probs: Action probabilities

        Returns:
            Entropy bonus
        """
        return self.entropy_coeff * self.compute_entropy(action_probs)


class CuriosityDrivenExploration:
    """Curiosity-driven exploration using prediction error"""

    def __init__(
        self,
        state_size: int,
        hidden_size: int = 64,
        lr: float = 0.001,
        beta: float = 0.2,
    ):
        """Initialize curiosity module

        Args:
            state_size: Dimension of state space
            hidden_size: Hidden layer size for forward model
            lr: Learning rate for forward model
            beta: Scaling factor for intrinsic reward
        """
        self.state_size = state_size
        self.beta = beta

        # Forward model: predicts next state from current state and action
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(state_size + 1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, state_size),
        )

        self.optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def compute_intrinsic_reward(
        self, state: np.ndarray, action: int, next_state: np.ndarray
    ) -> float:
        """Compute curiosity-driven intrinsic reward

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Intrinsic reward
        """
        # Prepare input for forward model
        state_action = np.concatenate([state, [action]])
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Predict next state
        predicted_next = self.forward_model(state_action_tensor)

        # Compute prediction error
        prediction_error = self.criterion(predicted_next, next_state_tensor).item()

        # Update forward model
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_next, next_state_tensor)
        loss.backward()
        self.optimizer.step()

        return self.beta * prediction_error


class ExplorationScheduler:
    """Adaptive exploration scheduling"""

    def __init__(self, strategy: str = "boltzmann", **kwargs):
        """Initialize exploration scheduler

        Args:
            strategy: Exploration strategy ('epsilon_greedy', 'boltzmann')
            **kwargs: Strategy-specific parameters
        """
        self.strategy_name = strategy

        if strategy == "epsilon_greedy":
            self.strategy = EpsilonGreedyExploration(**kwargs)
        elif strategy == "boltzmann":
            self.strategy = BoltzmannExploration(**kwargs)
        else:
            raise ValueError(f"Unknown exploration strategy: {strategy}")

        self.episode_rewards = []
        self.exploration_rates = []

    def select_action(self, action_probs: np.ndarray) -> int:
        """Select action using current exploration strategy

        Args:
            action_probs: Action probabilities from policy

        Returns:
            Selected action
        """
        return self.strategy.get_action(action_probs)

    def update_exploration(self, episode_reward: float, episode_done: bool = True):
        """Update exploration based on performance

        Args:
            episode_reward: Reward from episode
            episode_done: Whether episode ended
        """
        self.episode_rewards.append(episode_reward)

        # Store current exploration rate
        if hasattr(self.strategy, "epsilon"):
            self.exploration_rates.append(self.strategy.epsilon)
        elif hasattr(self.strategy, "temperature"):
            self.exploration_rates.append(self.strategy.temperature)

        # Adaptive update based on performance
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            if recent_avg > np.mean(self.episode_rewards):  # Improving
                # Reduce exploration more aggressively
                if hasattr(self.strategy, "decay_rate"):
                    self.strategy.decay_rate = min(
                        0.999, self.strategy.decay_rate * 1.01
                    )

        self.strategy.update(episode_done)

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration statistics

        Returns:
            Dictionary with exploration statistics
        """
        return {
            "strategy": self.strategy_name,
            "current_rate": (
                self.exploration_rates[-1] if self.exploration_rates else None
            ),
            "rate_history": self.exploration_rates,
            "total_episodes": len(self.episode_rewards),
            "avg_recent_reward": (
                np.mean(self.episode_rewards[-10:])
                if len(self.episode_rewards) >= 10
                else None
            ),
        }


class ExplorationVisualizer:
    """Visualize exploration behavior"""

    def __init__(self):
        """Initialize exploration visualizer"""
        pass

    def plot_exploration_schedule(
        self,
        exploration_rates: List[float],
        episode_rewards: List[float],
        title: str = "Exploration Schedule",
    ):
        """Plot exploration rate over time

        Args:
            exploration_rates: List of exploration rates
            episode_rewards: List of episode rewards
            title: Plot title
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        episodes = range(len(exploration_rates))

        axes[0].plot(episodes, exploration_rates, color="blue", linewidth=2)
        axes[0].set_title(f"{title} - Exploration Rate")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Exploration Rate")
        axes[0].grid(True, alpha=0.3)

        window = 20
        if len(episode_rewards) >= window:
            moving_avg = [
                np.mean(episode_rewards[i - window : i])
                for i in range(window, len(episode_rewards))
            ]
            axes[1].plot(
                range(window, len(episode_rewards)),
                moving_avg,
                color="red",
                linewidth=2,
                label=f"{window}-Episode Average",
            )

        axes[1].plot(
            episode_rewards, alpha=0.6, color="orange", label="Episode Rewards"
        )
        axes[1].set_title("Learning Progress")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Episode Reward")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_exploration_strategies(
        self,
        strategies_results: Dict[str, Dict],
        title: str = "Exploration Strategies Comparison",
    ):
        """Compare different exploration strategies

        Args:
            strategies_results: Dictionary of strategy results
            title: Plot title
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for strategy_name, results in strategies_results.items():
            rates = results.get("exploration_rates", [])
            rewards = results.get("episode_rewards", [])

            if rates:
                axes[0, 0].plot(rates, label=strategy_name, alpha=0.7)

            if rewards:
                axes[0, 1].plot(rewards, label=strategy_name, alpha=0.7)

                # Performance distribution
                axes[1, 0].boxplot(
                    [rewards],
                    positions=[list(strategies_results.keys()).index(strategy_name)],
                    labels=[strategy_name],
                )

        axes[0, 0].set_title("Exploration Rate Decay")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Exploration Rate")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Learning Curves")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Episode Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Performance Distribution")
        axes[1, 0].set_ylabel("Episode Reward")
        axes[1, 0].grid(True, alpha=0.3)

        # Sample efficiency comparison
        sample_efficiency = {}
        for strategy_name, results in strategies_results.items():
            rewards = results.get("episode_rewards", [])
            if rewards:
                # Episodes to reach 80% of max performance
                max_reward = np.max(rewards)
                threshold = 0.8 * max_reward
                episodes_to_threshold = next(
                    (i for i, r in enumerate(rewards) if r >= threshold), len(rewards)
                )
                sample_efficiency[strategy_name] = episodes_to_threshold

        if sample_efficiency:
            strategies = list(sample_efficiency.keys())
            episodes = list(sample_efficiency.values())

            bars = axes[1, 1].bar(
                strategies,
                episodes,
                alpha=0.7,
                color=["skyblue", "lightcoral", "lightgreen"],
            )
            axes[1, 1].set_title("Sample Efficiency (Episodes to 80% Performance)")
            axes[1, 1].set_ylabel("Episodes")
            axes[1, 1].grid(True, alpha=0.3)

            for bar, episodes in zip(bars, episodes):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(episodes)}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.show()


def create_exploration_strategy(strategy_name: str, **kwargs) -> ExplorationStrategy:
    """Factory function for exploration strategies

    Args:
        strategy_name: Name of exploration strategy
        **kwargs: Strategy parameters

    Returns:
        Exploration strategy instance
    """
    if strategy_name == "epsilon_greedy":
        return EpsilonGreedyExploration(**kwargs)
    elif strategy_name == "boltzmann":
        return BoltzmannExploration(**kwargs)
    else:
        raise ValueError(f"Unknown exploration strategy: {strategy_name}")


def test_exploration_strategy(
    strategy: ExplorationStrategy, n_tests: int = 1000
) -> Dict[str, Any]:
    """Test exploration strategy behavior

    Args:
        strategy: Exploration strategy to test
        n_tests: Number of test actions

    Returns:
        Test results dictionary
    """
    # Mock action probabilities (prefer action 0)
    action_probs = np.array([0.7, 0.2, 0.1])

    action_counts = np.zeros(len(action_probs))

    for _ in range(n_tests):
        action = strategy.get_action(action_probs)
        action_counts[action] += 1

    empirical_probs = action_counts / n_tests

    return {
        "action_counts": action_counts,
        "empirical_probabilities": empirical_probs,
        "theoretical_probabilities": action_probs,
        "exploration_rate": (
            1.0 - empirical_probs[0] / action_probs[0] if action_probs[0] > 0 else 0
        ),
    }
