import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import os
from typing import Any, Tuple, List, Union, Callable
import gymnasium as gym
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import NeuralModel, ModelTrainer, device
from environments.environments import SimpleGridWorld


class MPCController:
    """Model Predictive Control (MPC) controller for planning action sequences.

    This controller uses an environment model to simulate trajectories and optimize
    action sequences to minimize a cost function over a given horizon. It supports
    Cross-Entropy Method (CEM) and Random Shooting for optimization.

    Args:
        model (NeuralModel): The neural environment model used for prediction.
        num_actions (int): The number of discrete actions available.
        state_dim (int): The dimensionality of the state space.
        horizon (int, optional): The planning horizon (number of steps to look ahead).
                               Defaults to 10.
        num_samples (int, optional): Number of action sequences to sample per optimization iteration.
                                   Defaults to 100.
        cem_iterations (int, optional): Number of CEM iterations. Defaults to 5.
        elite_ratio (float, optional): The ratio of top-performing samples to select as elites in CEM.
                                     Defaults to 0.1.
    """

    def __init__(
        self,
        model: NeuralModel,
        num_actions: int,
        state_dim: int,
        horizon: int = 10,
        num_samples: int = 100,
        cem_iterations: int = 5,
        elite_ratio: float = 0.1,
    ):
        self.model = model
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.cem_iterations = cem_iterations
        self.elite_ratio = elite_ratio
        self.elite_size = max(1, int(num_samples * elite_ratio))

        self.optimization_costs: List[float] = []

    def cross_entropy_optimization(self, initial_state: int) -> np.ndarray:
        """Optimizes an action sequence using the Cross-Entropy Method (CEM).

        Args:
            initial_state (int): The starting state for the optimization.

        Returns:
            np.ndarray: The best action sequence found [horizon].
        """
        # Initialize action probabilities (uniform categorical distribution)
        # action_probs[h, a] is the probability of taking action `a` at horizon step `h`
        action_probs = np.ones((self.horizon, self.num_actions), dtype=np.float32) / self.num_actions

        best_cost = float("inf")
        best_actions_sequence = np.zeros(self.horizon, dtype=int)

        for iteration in range(self.cem_iterations):
            action_sequences: List[List[int]] = []
            costs: List[float] = []

            for _ in range(self.num_samples):
                current_sequence: List[int] = []
                for h in range(self.horizon):
                    # Sample action from current categorical distribution
                    action = np.random.choice(self.num_actions, p=action_probs[h])
                    current_sequence.append(action)

                action_sequences.append(current_sequence)
                cost = self.evaluate_sequence(initial_state, current_sequence)
                costs.append(cost)

            # Select elite sequences based on lowest cost
            elite_indices = np.argsort(costs)[: self.elite_size]
            elite_sequences = [action_sequences[i] for i in elite_indices]

            # Update action probabilities based on elite sequences
            new_action_probs = np.zeros((self.horizon, self.num_actions), dtype=np.float32)
            for h in range(self.horizon):
                for seq in elite_sequences:
                    new_action_probs[h, seq[h]] += 1

            # Normalize and smooth the probabilities
            # Add a small epsilon to avoid division by zero and maintain some exploration
            new_action_probs = (new_action_probs + 1e-8) / (self.elite_size + self.num_actions * 1e-8) # Laplace smoothing
            action_probs = new_action_probs # Direct update for simplicity in this CEM version

            min_cost_this_iter = np.min(costs)
            if min_cost_this_iter < best_cost:
                best_cost = min_cost_this_iter
                best_actions_sequence = np.array(action_sequences[np.argmin(costs)], dtype=int)

        self.optimization_costs.append(best_cost) # Store only the final best cost for the entire CEM run
        return best_actions_sequence

    def random_shooting(self, initial_state: int) -> np.ndarray:
        """Performs Random Shooting optimization to find a good action sequence.

        Args:
            initial_state (int): The starting state for the optimization.

        Returns:
            np.ndarray: The best action sequence found [horizon].
        """
        best_cost = float("inf")
        best_actions_sequence = np.zeros(self.horizon, dtype=int)

        for _ in range(self.num_samples):
            actions = [np.random.randint(self.num_actions) for _ in range(self.horizon)]

            cost = self.evaluate_sequence(initial_state, actions)

            if cost < best_cost:
                best_cost = cost
                best_actions_sequence = np.array(actions, dtype=int)

        self.optimization_costs.append(best_cost)
        return best_actions_sequence

    def evaluate_sequence(self, initial_state: int, actions: List[int]) -> float:
        """Evaluates the cumulative cost of an action sequence using the environment model.

        The cost is defined as the negative of the reward, aiming to minimize cost.

        Args:
            initial_state (int): The starting state for simulating the sequence.
            actions (List[int]): The sequence of actions to evaluate [horizon].

        Returns:
            float: The total discounted cost (negative reward) of the sequence.
        """
        state = initial_state
        total_cost = 0.0
        discount = 1.0
        gamma_discount = 0.95 # Consistent discount factor

        # Convert initial state to one-hot tensor if it's discrete
        current_state_tensor = torch.eye(self.state_dim)[state:state+1].to(device) # [1, state_dim]

        for action in actions:
            action_tensor = torch.tensor([action], dtype=torch.long).to(device) # [1,]

            # Model expects state [batch_size, state_dim], action [batch_size, action_dim] (one-hot)
            # Or action [batch_size,] if discrete. NeuralModel handles conversion.
            next_state_pred_logits, reward_pred_scalar = self.model.forward(
                current_state_tensor, action_tensor
            )

            # Convert predicted next state logits to discrete index
            next_state = torch.argmax(next_state_pred_logits.squeeze(0)).item()
            reward = reward_pred_scalar.squeeze(0).item()

            cost = -reward # Cost is negative of reward
            total_cost += discount * cost
            discount *= gamma_discount

            state = next_state
            # Update current_state_tensor for next step in horizon
            current_state_tensor = torch.eye(self.state_dim)[state:state+1].to(device)

        return total_cost

    def select_action(self, state: int, method: str = "cross_entropy") -> int:
        """Selects the best immediate action using MPC planning.

        Args:
            state (int): The current state.
            method (str, optional): The optimization method to use ("cross_entropy" or "random_shooting").
                                  Defaults to "cross_entropy".

        Returns:
            int: The first action of the optimal sequence found.
        """
        action_sequence: np.ndarray
        if method == "cross_entropy":
            action_sequence = self.cross_entropy_optimization(state)
        elif method == "random_shooting":
            action_sequence = self.random_shooting(state)
        else:
            raise ValueError(f"Unknown MPC optimization method: {method}")

        return action_sequence[0] if action_sequence.size > 0 else np.random.randint(self.num_actions)


class MPCAgent:
    """Reinforcement Learning Agent that uses Model Predictive Control (MPC) for action selection.

    Args:
        model (NeuralModel): The neural environment model used by the MPC controller.
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions available to the agent.
        horizon (int, optional): The planning horizon for the MPC controller. Defaults to 10.
        method (str, optional): The MPC optimization method ("cross_entropy" or "random_shooting").
                               Defaults to "cross_entropy".
    """

    def __init__(
        self,
        model: NeuralModel,
        num_states: int,
        num_actions: int,
        horizon: int = 10,
        method: str = "cross_entropy",
    ):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.controller = MPCController(model, num_actions, num_states, horizon=horizon)
        self.method = method

        self.episode_rewards: List[float] = []
        self.planning_costs: List[float] = []

    def train_episode(
        self, env: SimpleGridWorld, max_steps: int = 200
    ) -> Tuple[float, int]:
        """Runs a single episode, with the agent selecting actions using MPC planning.

        Args:
            env (SimpleGridWorld): The environment to interact with.
            max_steps (int, optional): Maximum steps per episode. Defaults to 200.

        Returns:
            Tuple[float, int]: A tuple containing the total reward and number of steps in the episode.
        """
        state, _ = env.reset() # Gymnasium API
        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            action = self.controller.select_action(state, self.method)
            next_state, reward, terminated, truncated, _ = env.step(action) # Gymnasium API
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.episode_rewards.append(total_reward)
        # Collect all optimization costs incurred during the episode
        self.planning_costs.extend(self.controller.optimization_costs)
        self.controller.optimization_costs.clear() # Clear for next episode

        return total_reward, steps

    def get_statistics(self) -> Dict[str, Union[float, int]]:
        """Retrieves performance statistics for the MPC agent.

        Returns:
            Dict[str, Union[float, int]]: A dictionary containing:
                - 'avg_episode_reward' (float): Average reward over the last 10 episodes.
                - 'avg_planning_cost' (float): Average cost incurred during MPC optimization.
                - 'total_episodes' (int): Total number of episodes run.
        """
        return {
            "avg_episode_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            ),
            "avg_planning_cost": (
                np.mean(self.planning_costs) if self.planning_costs else 0.0 # Average over all collected costs
            ),
            "total_episodes": len(self.episode_rewards),
        }


def demonstrate_mpc():
    """Demonstrates the Model Predictive Control (MPC) algorithm.

    This function sets up a `SimpleGridWorld` environment, trains a `NeuralModel`
    to learn environment dynamics, and then uses `MPCAgent` with both Cross-Entropy Method (CEM)
    and Random Shooting (RS) for planning. It visualizes the agents' performance,
    planning costs, and analyzes the impact of planning horizon.
    """
    print("\n--- Model Predictive Control (MPC) Demonstration ---")
    print("=" * 50)

    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("\n1. Setting up environment and training Neural Model...")
    env = SimpleGridWorld(size=5)
    num_states = env.num_states
    num_actions = env.num_actions

    # Collect experience using random policy for neural model training
    n_experience_episodes = 1000
    experience_data: List[Tuple[int, int, int, float]] = []

    print(f"  Collecting {n_experience_episodes} episodes of random experience for model training...")
    for episode in range(n_experience_episodes):
        state, _ = env.reset() # Gymnasium API
        done = False

        while not done:
            action = np.random.randint(num_actions)  # Random policy
            next_state, reward, terminated, truncated, _ = env.step(action) # Gymnasium API
            done = terminated or truncated

            experience_data.append((state, action, next_state, reward))

            state = next_state
    print(f"  Collected {len(experience_data)} transitions.")

    # Prepare data for neural model training
    states_np = np.array([exp[0] for exp in experience_data], dtype=np.int64)
    actions_np = np.array([exp[1] for exp in experience_data], dtype=np.int64)
    next_states_np = np.array([exp[2] for exp in experience_data], dtype=np.int64)
    rewards_np = np.array([exp[3] for exp in experience_data], dtype=np.float32)

    # Convert states to one-hot for NeuralModel input (expects float32)
    states_onehot_np = np.eye(num_states)[states_np].astype(np.float32)
    next_states_onehot_np = np.eye(num_states)[next_states_np].astype(np.float32)

    # Train neural model using ModelTrainer
    neural_model = NeuralModel(num_states, num_actions, hidden_dim=64, ensemble_size=5).to(device)
    trainer = ModelTrainer(neural_model, lr=1e-3)

    print("  Training neural model for MPC...")
    trainer.train_batch(
        (states_onehot_np, actions_np, next_states_onehot_np, rewards_np), epochs=50, batch_size=64
    )

    # 2. Testing MPC performance with different optimization methods
    print("\n2. Testing MPC performance with Cross-Entropy Method (CEM) and Random Shooting (RS)...")
    agents: Dict[str, MPCAgent] = {
        "MPC-CEM": MPCAgent(
            neural_model,
            num_states,
            num_actions,
            horizon=8,
            method="cross_entropy",
        ),
        "MPC-RS": MPCAgent(
            neural_model,
            num_states,
            num_actions,
            horizon=8,
            method="random_shooting",
        ),
    }

    n_test_episodes = 15
    results: Dict[str, Dict[str, Any]] = {}

    for name, agent in agents.items():
        print(f"  Testing {name}...")
        # Reset agent's internal state for each run
        agent.episode_rewards = []
        agent.planning_costs = []
        # Reset controller's costs too
        agent.controller.optimization_costs.clear()

        for episode in range(n_test_episodes):
            reward, length = agent.train_episode(env, max_steps=100)

            if (episode + 1) % 5 == 0 or episode == n_test_episodes - 1:
                avg_reward = np.mean(agent.episode_rewards[-5:]) if len(agent.episode_rewards) >= 5 else np.mean(agent.episode_rewards)
                print(
                    f"    Episodes {max(0, episode-4)}-{episode+1}: Avg Reward = {avg_reward:.2f}"
                )

        results[name] = {
            "episode_rewards": agent.episode_rewards.copy(),
            "episode_lengths": agent.episode_lengths.copy(),
            "statistics": agent.get_statistics(),
        }

    print("\n3. Generating Visualizations...")
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    for name, data in results.items():
        rewards = np.array(data["episode_rewards"])
        smoothed_rewards = pd.Series(rewards).rolling(window=3, min_periods=1).mean()
        plt.plot(smoothed_rewards, linewidth=2, label=name, marker="o", markersize=4)

    plt.title("MPC Performance Comparison (Smoothed Rewards)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    for name, data in results.items():
        lengths = np.array(data["episode_lengths"])
        smoothed_lengths = pd.Series(lengths).rolling(window=3, min_periods=1).mean()
        plt.plot(smoothed_lengths, linewidth=2, label=name, marker="s", markersize=4)

    plt.title("Episode Lengths (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Goal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    reward_data_for_boxplot = [results[name]["episode_rewards"] for name in results.keys()]
    labels_for_boxplot = list(results.keys())
    plt.boxplot(reward_data_for_boxplot, labels=labels_for_boxplot)
    plt.title("Reward Distribution Across Methods")
    plt.ylabel("Episode Reward")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    if "MPC-CEM" in results:
        cem_agent_costs = results["MPC-CEM"]["statistics"]["avg_planning_cost"]
        if cem_agent_costs > 0: # Only plot if there are actual costs
            # Note: optimization_costs list in controller gets cleared per episode.
            # We need to collect per-episode planning costs separately for plotting historical.
            # For now, this plot will show the *average* planning cost for the CEM agent.
            plt.bar(["MPC-CEM"], [cem_agent_costs], color="purple", alpha=0.7)
            plt.title("Average Planning Cost (MPC-CEM)")
            plt.ylabel("Average Cost")
            plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    print("\n  Analyzing effect of planning horizon...")
    horizon_results: Dict[int, float] = {}
    horizons = [3, 5, 8, 12]

    for h in horizons:
        print(f"    Testing with horizon = {h}...")
        # Reset environment state for each horizon test
        env_reset_state, _ = env.reset()
        agent = MPCAgent(
            neural_model,
            num_states,
            num_actions,
            horizon=h,
            method="cross_entropy",
        )
        rewards = []

        n_horizon_test_episodes = 5 # Small number of episodes for quick test
        for episode in range(n_horizon_test_episodes):
            # Need to re-reset env for each episode in horizon test
            current_env_for_test = SimpleGridWorld(size=5) # Re-instantiate if env state is problematic
            reward, _ = agent.train_episode(current_env_for_test, max_steps=100)
            rewards.append(reward)

        horizon_results[h] = np.mean(rewards)
    print("  Horizon analysis complete.")

    horizons_list = list(horizon_results.keys())
    performance_list = list(horizon_results.values())
    plt.bar(
        horizons_list, performance_list, alpha=0.7, color="skyblue", edgecolor="black", width=1.5
    )
    plt.title("Performance vs Planning Horizon")
    plt.xlabel("Planning Horizon")
    plt.ylabel("Average Episode Reward")
    plt.xticks(horizons_list) # Ensure all horizons are shown as ticks
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    method_names = list(results.keys())
    avg_rewards = [np.mean(results[name]["episode_rewards"]) for name in method_names]
    avg_lengths = [np.mean(results[name]["episode_lengths"]) for name in method_names]

    x = np.arange(len(method_names))
    width = 0.35

    plt.bar(x - width / 2, avg_rewards, width, label="Avg Reward", alpha=0.7)
    plt.bar(
        x + width / 2,
        [l / 10.0 for l in avg_lengths], # Scale length for better visualization if values are too large
        width,
        label="Avg Length / 10",
        alpha=0.7,
    )

    plt.title("MPC Method Performance Comparison")
    plt.xlabel("Method")
    plt.ylabel("Scaled Performance / Length")
    plt.xticks(x, method_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/mpc_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n4. MPC Analysis Summary:")
    for name, data in results.items():
        stats = data["statistics"]
        # Handle potential empty episode_rewards for std calculation
        std_reward = np.std(data['episode_rewards']) if len(data['episode_rewards']) > 1 else 0.0
        std_length = np.std(data['episode_lengths']) if len(data['episode_lengths']) > 1 else 0.0

        print(f"\n  {name}:")
        print(
            f"    Average Episode Reward: {np.mean(data['episode_rewards']):.3f} ± {std_reward:.3f}"
        )
        print(
            f"    Average Episode Length: {np.mean(data['episode_lengths']):.1f} ± {std_length:.1f}"
        )
        if stats["avg_planning_cost"] > 0:
            print(f"    Average Planning Cost: {stats['avg_planning_cost']:.3f}")
        else:
            print(f"    Average Planning Cost: Not available (or zero)")

    print(f"\n  Horizon Analysis:")
    for h, perf in horizon_results.items():
        print(f"    Horizon {h}: {perf:.3f} average reward")

    print("\n📊 Key MPC Insights:")
    print("• MPC provides principled planning with explicit horizons.")
    print("• Cross-Entropy Method generally achieves better performance than random shooting for the same number of samples.")
    print("• Longer planning horizons can improve performance by allowing more look-ahead, but at the cost of increased computational complexity.")
    print("• MPC is well-suited for continuous control and discrete planning problems where a model of the dynamics is available.")
    print("• The cost function design is crucial for effective MPC performance.")

    print("\n✅ Model Predictive Control (MPC) demonstration complete!")
    print("📊 Check 'visualizations/mpc_analysis.png' for plots.")
    print("""
All core Python files reviewed and documentation generated.
Next steps: review remaining utility and evaluation files if necessary, then finalize.
""")

if __name__ == "__main__":
    demonstrate_mpc()
