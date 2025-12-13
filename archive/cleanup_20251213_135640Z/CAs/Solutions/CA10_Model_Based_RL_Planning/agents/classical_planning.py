import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import os
from typing import Any, Tuple, List, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import TabularModel, NeuralModel, ModelTrainer, device
from environments.environments import SimpleGridWorld


class ModelBasedPlanner:
    """Classical planning algorithms using learned models.

    This class implements Value Iteration and Policy Iteration, which compute optimal
    value functions and policies given a model of the environment.

    Args:
        model (TabularModel): The environment model (e.g., TabularModel) that provides
                              transition probabilities and expected rewards.
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions available to the agent.
        gamma (float, optional): The discount factor. Defaults to 0.99.
    """

    def __init__(
        self, model: TabularModel, num_states: int, num_actions: int, gamma: float = 0.99
    ):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

        # Initialize value function and policy
        self.V = np.zeros(num_states, dtype=np.float32)
        self.policy = np.zeros(num_states, dtype=int)

        # Planning history for analysis
        self.value_history: List[np.ndarray] = []
        self.policy_history: List[np.ndarray] = []

    def value_iteration(
        self, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Value Iteration using the learned environment model.

        Args:
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - V (np.ndarray): The converged optimal value function [num_states].
                - policy (np.ndarray): The optimal policy [num_states].
        """
        print(f"Running Value Iteration (max_iter={max_iterations}, tol={tolerance})")

        for iteration in range(max_iterations):
            old_V = self.V.copy()

            for state in range(self.num_states):
                q_values = np.zeros(self.num_actions, dtype=np.float32)

                for action in range(self.num_actions):
                    expected_value = 0.0

                    for next_state in range(self.num_states):
                        transition_prob = self.model.get_transition_prob(
                            state, action, next_state
                        )
                        reward = self.model.get_expected_reward(state, action)
                        expected_value += transition_prob * (
                            reward + self.gamma * old_V[next_state]
                        )

                    q_values[action] = expected_value

                self.V[state] = np.max(q_values)
                self.policy[state] = np.argmax(q_values)

            self.value_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())

            if np.max(np.abs(self.V - old_V)) < tolerance:
                print(f"Value Iteration converged after {iteration + 1} iterations")
                break
        else:
            print(f"Value Iteration finished after {max_iterations} iterations without full convergence.")

        return self.V, self.policy

    def policy_iteration(
        self, max_iterations: int = 50, eval_max_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Policy Iteration using the learned environment model.

        Args:
            max_iterations (int, optional): Maximum number of policy improvement iterations. Defaults to 50.
            eval_max_iterations (int, optional): Maximum iterations for policy evaluation. Defaults to 100.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - V (np.ndarray): The converged value function for the optimal policy [num_states].
                - policy (np.ndarray): The optimal policy [num_states].
        """
        print(f"Running Policy Iteration (max_iter={max_iterations})")

        # Initialize random policy
        self.policy = np.random.randint(0, self.num_actions, self.num_states)

        for iteration in range(max_iterations):
            old_policy = self.policy.copy()

            # Policy Evaluation
            self.V = self.policy_evaluation(
                self.policy, max_iterations=eval_max_iterations
            )

            # Policy Improvement
            for state in range(self.num_states):
                q_values = np.zeros(self.num_actions, dtype=np.float32)

                for action in range(self.num_actions):
                    expected_value = 0.0

                    for next_state in range(self.num_states):
                        transition_prob = self.model.get_transition_prob(
                            state, action, next_state
                        )
                        reward = self.model.get_expected_reward(state, action)
                        expected_value += transition_prob * (
                            reward + self.gamma * self.V[next_state]
                        )

                    q_values[action] = expected_value

                self.policy[state] = np.argmax(q_values)

            self.value_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())

            if np.array_equal(self.policy, old_policy):
                print(f"Policy Iteration converged after {iteration + 1} iterations")
                break
        else:
            print(f"Policy Iteration finished after {max_iterations} iterations without full convergence.")

        return self.V, self.policy

    def policy_evaluation(
        self, policy: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> np.ndarray:
        """Evaluates a given policy to compute its state-value function.

        Args:
            policy (np.ndarray): The policy to evaluate [num_states].
            max_iterations (int, optional): Maximum number of evaluation iterations. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.

        Returns:
            np.ndarray: The value function for the given policy [num_states].
        """
        V = np.zeros(self.num_states, dtype=np.float32)

        for iteration in range(max_iterations):
            old_V = V.copy()

            for state in range(self.num_states):
                action = policy[state]
                expected_value = 0.0

                for next_state in range(self.num_states):
                    transition_prob = self.model.get_transition_prob(
                        state, action, next_state
                    )
                    reward = self.model.get_expected_reward(state, action)
                    expected_value += transition_prob * (
                        reward + self.gamma * old_V[next_state]
                    )

                V[state] = expected_value

            if np.max(np.abs(V - old_V)) < tolerance:
                break

        return V

    def compute_q_function(self) -> np.ndarray:
        """Computes the Q-function from the current value function V.

        Returns:
            np.ndarray: The Q-function [num_states, num_actions].
        """
        Q = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        for state in range(self.num_states):
            for action in range(self.num_actions):
                expected_value = 0.0

                for next_state in range(self.num_states):
                    transition_prob = self.model.get_transition_prob(
                        state, action, next_state
                    )
                    reward = self.model.get_expected_reward(state, action)
                    expected_value += transition_prob * (
                        reward + self.gamma * self.V[next_state]
                    )

                Q[state, action] = expected_value

        return Q


class UncertaintyAwarePlanner:
    """Planner that incorporates model uncertainty for robust decision-making.

    This class uses an ensemble neural model to estimate pessimistic or optimistic
    rewards to guide value iteration.

    Args:
        ensemble_model (NeuralModel): A neural model ensemble that can provide
                                      predictions with uncertainty (mean and std).
        num_states (int): The number of states in the environment.
        num_actions (int): The number of actions available to the agent.
        gamma (float, optional): The discount factor. Defaults to 0.99.
    """

    def __init__(
        self, ensemble_model: NeuralModel, num_states: int, num_actions: int, gamma: float = 0.99
    ):
        self.ensemble_model = ensemble_model
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

    def pessimistic_value_iteration(
        self, beta: float = 1.0, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Value Iteration using pessimistic model estimates.

        Pessimistic estimates subtract `beta` times the standard deviation from the mean reward.

        Args:
            beta (float, optional): Uncertainty scaling factor. Higher beta means more pessimism.
                                    Defaults to 1.0.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - V (np.ndarray): The converged value function under pessimistic planning [num_states].
                - policy (np.ndarray): The derived policy under pessimistic planning [num_states].
        """
        V = np.zeros(self.num_states, dtype=np.float32)
        policy = np.zeros(self.num_states, dtype=int)

        print(f"Running Pessimistic Value Iteration (beta={beta}, max_iter={max_iterations})")

        self.ensemble_model.eval()  # Set model to eval mode

        for iteration in range(max_iterations):
            old_V = V.copy()

            for state in range(self.num_states):
                q_values = np.zeros(self.num_actions, dtype=np.float32)

                for action in range(self.num_actions):
                    # Prepare state and action tensors for NeuralModel
                    # NeuralModel expects state as float32 and discrete action as long
                    state_tensor = torch.eye(self.num_states)[state : state + 1].to(device)
                    action_tensor = torch.tensor([action], dtype=torch.long).to(device)

                    # Get ensemble predictions with uncertainty
                    next_state_mean, reward_mean, next_state_std, reward_std = (
                        self.ensemble_model.predict_with_uncertainty(
                            state_tensor, action_tensor
                        )
                    )

                    # Use pessimistic reward estimate (mean - beta * std)
                    pessimistic_reward = (
                        reward_mean.cpu().item() - beta * reward_std.cpu().item()
                    )

                    # For simplicity, use the most likely next state for transition
                    # next_state_idx = torch.argmax(next_state_mean.squeeze(0)).item() # For one-hot output
                    # Assuming next_state_mean is a vector of probabilities/logits for next state
                    next_state_idx = np.argmax(next_state_mean.cpu().numpy().squeeze()) # Get index of most likely next state

                    q_values[action] = (
                        pessimistic_reward + self.gamma * old_V[next_state_idx]
                    )

                V[state] = np.max(q_values)
                policy[state] = np.argmax(q_values)

            if np.max(np.abs(V - old_V)) < tolerance:
                print(f"Pessimistic VI converged after {iteration + 1} iterations")
                break
        else:
            print(f"Pessimistic VI finished after {max_iterations} iterations without full convergence.")

        return V, policy

    def optimistic_value_iteration(
        self, beta: float = 1.0, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs Value Iteration using optimistic model estimates.

        Optimistic estimates add `beta` times the standard deviation to the mean reward.

        Args:
            beta (float, optional): Uncertainty scaling factor. Higher beta means more optimism.
                                    Defaults to 1.0.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-6.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - V (np.ndarray): The converged value function under optimistic planning [num_states].
                - policy (np.ndarray): The derived policy under optimistic planning [num_states].
        """
        V = np.zeros(self.num_states, dtype=np.float32)
        policy = np.zeros(self.num_states, dtype=int)

        print(f"Running Optimistic Value Iteration (beta={beta}, max_iter={max_iterations})")

        self.ensemble_model.eval()  # Set model to eval mode

        for iteration in range(max_iterations):
            old_V = V.copy()

            for state in range(self.num_states):
                q_values = np.zeros(self.num_actions, dtype=np.float32)

                for action in range(self.num_actions):
                    # Prepare state and action tensors for NeuralModel
                    state_tensor = torch.eye(self.num_states)[state : state + 1].to(device)
                    action_tensor = torch.tensor([action], dtype=torch.long).to(device)

                    # Get ensemble predictions with uncertainty
                    next_state_mean, reward_mean, next_state_std, reward_std = (
                        self.ensemble_model.predict_with_uncertainty(
                            state_tensor, action_tensor
                        )
                    )

                    # Use optimistic reward estimate (mean + beta * std)
                    optimistic_reward = (
                        reward_mean.cpu().item() + beta * reward_std.cpu().item()
                    )

                    # For simplicity, use the most likely next state for transition
                    next_state_idx = np.argmax(next_state_mean.cpu().numpy().squeeze())

                    q_values[action] = (
                        optimistic_reward + self.gamma * old_V[next_state_idx]
                    )

                V[state] = np.max(q_values)
                policy[state] = np.argmax(q_values)

            if np.max(np.abs(V - old_V)) < tolerance:
                print(f"Optimistic VI converged after {iteration + 1} iterations")
                break
        else:
            print(f"Optimistic VI finished after {max_iterations} iterations without full convergence.")

        return V, policy


class ModelBasedPolicySearch:
    """Policy search algorithms that directly search for a good policy using a learned model.

    Args:
        model (Union[TabularModel, NeuralModel]): The environment model.
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        gamma (float, optional): Discount factor. Defaults to 0.99.
    """

    def __init__(
        self, model: Union[TabularModel, NeuralModel], state_dim: int, action_dim: int, gamma: float = 0.99
    ):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

    def random_shooting(
        self, initial_state: int, horizon: int = 10, num_sequences: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """Performs random shooting to find a good action sequence.

        Args:
            initial_state (int): The starting state for planning.
            horizon (int, optional): The planning horizon. Defaults to 10.
            num_sequences (int, optional): Number of random action sequences to sample. Defaults to 1000.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - best_sequence (np.ndarray): The best action sequence found [horizon].
                - best_value (float): The cumulative discounted reward for the best sequence.
        """
        best_sequence = np.zeros(horizon, dtype=int)
        best_value = -np.inf

        for _ in range(num_sequences):
            action_sequence = np.random.randint(0, self.action_dim, horizon)

            total_reward = 0.0
            current_state = initial_state
            discount = 1.0

            for action in action_sequence:
                # Handle state for neural model (one-hot if discrete)
                if isinstance(self.model, NeuralModel):
                    state_tensor = torch.eye(self.state_dim)[current_state:current_state+1].to(device)
                    action_tensor = torch.tensor([action], dtype=torch.long).to(device)
                    next_state_pred, reward_pred = self.model.sample_transition(state_tensor, action_tensor)
                    next_state = np.argmax(next_state_pred.cpu().numpy().squeeze())
                    reward = reward_pred.cpu().item()
                else: # TabularModel
                    next_state, reward = self.model.sample_transition(current_state, action)
                
                total_reward += discount * reward
                discount *= self.gamma
                current_state = next_state

            if total_reward > best_value:
                best_value = total_reward
                best_sequence = action_sequence

        return best_sequence, best_value

    def cross_entropy_method(
        self,
        initial_state: int,
        horizon: int = 10,
        num_sequences: int = 1000,
        num_elite: int = 100,
        num_iterations: int = 10,
    ) -> Tuple[np.ndarray, float]:
        """Performs Cross-Entropy Method (CEM) for policy search.

        Args:
            initial_state (int): The starting state for planning.
            horizon (int, optional): The planning horizon. Defaults to 10.
            num_sequences (int, optional): Number of random action sequences to sample per iteration.
                                          Defaults to 1000.
            num_elite (int, optional): Number of best (elite) sequences to select. Defaults to 100.
            num_iterations (int, optional): Number of CEM iterations. Defaults to 10.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - best_sequence (np.ndarray): The best action sequence found [horizon].
                - best_value (float): The cumulative discounted reward for the best sequence.
        """
        # Initialize action probabilities (uniform)
        action_probs = np.ones((horizon, self.action_dim)) / self.action_dim

        print(f"Running Cross-Entropy Method (horizon={horizon}, iterations={num_iterations})")

        for iteration in range(num_iterations):
            sequences = []
            values = []

            for _ in range(num_sequences):
                sequence = []
                for t in range(horizon):
                    action = np.random.choice(self.action_dim, p=action_probs[t])
                    sequence.append(action)

                # Evaluate sequence using model
                total_reward = 0.0
                current_state = initial_state
                discount = 1.0

                for action in sequence:
                    # Handle state for neural model (one-hot if discrete)
                    if isinstance(self.model, NeuralModel):
                        state_tensor = torch.eye(self.state_dim)[current_state:current_state+1].to(device)
                        action_tensor = torch.tensor([action], dtype=torch.long).to(device)
                        next_state_pred, reward_pred = self.model.sample_transition(state_tensor, action_tensor)
                        next_state = np.argmax(next_state_pred.cpu().numpy().squeeze())
                        reward = reward_pred.cpu().item()
                    else: # TabularModel
                        next_state, reward = self.model.sample_transition(current_state, action)

                    total_reward += discount * reward
                    discount *= self.gamma
                    current_state = next_state

                sequences.append(sequence)
                values.append(total_reward)

            # Select elite sequences
            elite_indices = np.argsort(values)[-num_elite:]
            elite_sequences = [sequences[i] for i in elite_indices]

            # Update action probabilities
            action_counts = np.zeros((horizon, self.action_dim), dtype=np.float32)

            for sequence in elite_sequences:
                for t, action in enumerate(sequence):
                    action_counts[t, action] += 1

            # Smooth update
            alpha = 0.7
            new_probs = action_counts / num_elite
            action_probs = alpha * new_probs + (1 - alpha) * action_probs

            # Add small amount of noise for exploration
            action_probs += 1e-6 # Add a small epsilon to avoid zero probabilities
            action_probs /= np.sum(action_probs, axis=1, keepdims=True)

        # Return best sequence from the last iteration's elite set
        best_sequence_idx = np.argmax([values[i] for i in elite_indices])
        best_sequence = elite_sequences[best_sequence_idx]
        best_value = values[elite_indices[best_sequence_idx]]

        return np.array(best_sequence, dtype=int), best_value


def demonstrate_classical_planning():
    """Demonstrates classical planning algorithms (Value Iteration, Policy Iteration)
    and uncertainty-aware planning using learned environment models.

    This function collects experience, trains both tabular and neural models,
    then applies various planning algorithms and visualizes their results.
    """
    print("\n--- Classical Planning with Learned Models Demonstration ---")
    print("=" * 60)

    # Create environment and collect data
    env = SimpleGridWorld(size=4)
    num_states = env.num_states
    num_actions = env.num_actions

    tabular_model = TabularModel(num_states, num_actions)

    # Collect experience using random policy
    n_episodes = 200
    experience_data: List[Tuple[int, int, int, float]] = []

    print("\n1. Collecting experience for model learning (Random Policy)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.random.randint(num_actions)  # Random policy
            next_state, reward, terminated, truncated, _ = env.step(action) # Gymnasium API
            done = terminated or truncated

            # Update tabular model
            tabular_model.update(state, action, next_state, reward)

            # Store for neural model training
            experience_data.append((state, action, next_state, reward))

            state = next_state

    print(f"Collected {len(experience_data)} transitions.")

    # Prepare data for neural model training
    states_np = np.array([exp[0] for exp in experience_data], dtype=np.int64)
    actions_np = np.array([exp[1] for exp in experience_data], dtype=np.int64)
    next_states_np = np.array([exp[2] for exp in experience_data], dtype=np.int64)
    rewards_np = np.array([exp[3] for exp in experience_data], dtype=np.float32)

    # Convert states to one-hot for neural model input (expects float32)
    states_onehot_np = np.eye(num_states)[states_np].astype(np.float32)
    next_states_onehot_np = np.eye(num_states)[next_states_np].astype(np.float32)

    # Train neural model using ModelTrainer
    print("\n2. Training Neural Model (Ensemble) from collected experience...")
    neural_model = NeuralModel(
        num_states, num_actions, hidden_dim=64, ensemble_size=5
    ).to(device)
    trainer = ModelTrainer(neural_model, lr=1e-3)

    trainer.train_batch(
        (states_onehot_np, actions_np, next_states_onehot_np, rewards_np), epochs=20, batch_size=64
    )

    # 3. Classical Planning with Tabular Model
    print("\n3. Running Classical Planning with the Tabular Model...")
    planner = ModelBasedPlanner(tabular_model, num_states, num_actions, gamma=0.95)

    print("  Value Iteration:")
    vi_values, vi_policy = planner.value_iteration(max_iterations=50)

    print("  Policy Iteration:")
    # Re-initialize planner to reset V and policy for fair comparison for policy iteration
    planner_pi = ModelBasedPlanner(tabular_model, num_states, num_actions, gamma=0.95)
    pi_values, pi_policy = planner_pi.policy_iteration(max_iterations=20)

    # 4. Uncertainty-Aware Planning with Neural Model
    print("\n4. Running Uncertainty-Aware Planning with the Neural Model (Ensemble)...")
    uncertainty_planner = UncertaintyAwarePlanner(
        neural_model, num_states, num_actions, gamma=0.95
    )
    print("  Pessimistic Value Iteration:")
    pessimistic_V, pessimistic_policy = uncertainty_planner.pessimistic_value_iteration(
        beta=0.5, max_iterations=50
    )
    print("  Optimistic Value Iteration:")
    optimistic_V, optimistic_policy = uncertainty_planner.optimistic_value_iteration(
        beta=0.5, max_iterations=50
    )

    # 5. Visualization
    print("\n5. Generating Visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle("Classical Planning & Uncertainty-Aware Planning Results", fontsize=16)

    grid_size = int(np.sqrt(num_states))

    def plot_value_function(ax: plt.Axes, values: np.ndarray, title: str):
        value_grid = values.reshape(grid_size, grid_size)
        im = ax.imshow(value_grid, cmap="viridis")
        ax.set_title(title)
        # Add colorbar only once per row for cleaner visualization
        if ax.get_subplotspec().is_first_col():
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def plot_policy(ax: plt.Axes, policy: np.ndarray, title: str):
        policy_grid = policy.reshape(grid_size, grid_size)
        arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

        ax.imshow(np.zeros((grid_size, grid_size)), cmap="gray", alpha=0.1)

        for i in range(grid_size):
            for j in range(grid_size):
                state_idx = i * grid_size + j
                if state_idx in env.terminal_states:
                    ax.text(j, i, "G", ha="center", va="center", fontsize=20, color="green", fontweight="bold")
                elif state_idx in env.obstacle_states:
                    ax.text(j, i, "X", ha="center", va="center", fontsize=20, color="red", fontweight="bold")
                else:
                    action = policy_grid[i, j]
                    ax.text(
                        j,
                        i,
                        arrow_map[action],
                        ha="center",
                        va="center",
                        fontsize=18,
                        color="blue",
                        fontweight="bold",
                    )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot value functions
    plot_value_function(axes[0, 0], vi_values, "Value Iteration (Tabular) - Values")
    plot_value_function(axes[0, 1], pi_values, "Policy Iteration (Tabular) - Values")

    plot_value_function(axes[1, 0], pessimistic_V, "Pessimistic VI (Neural) - Values")
    plot_value_function(axes[1, 1], optimistic_V, "Optimistic VI (Neural) - Values")

    # Plot policies
    plot_policy(axes[2, 0], vi_policy, "Value Iteration (Tabular) - Policy")
    plot_policy(axes[2, 1], pi_policy, "Policy Iteration (Tabular) - Policy")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/classical_planning.png", dpi=300, bbox_inches="tight")
    plt.show() # Only show once at the end

    # 6. Compare planning methods & Policy Search
    print("\n6. Planning Method Comparison & Policy Search:")
    print(
        f"  Value Iteration - Max Value: {np.max(vi_values):.3f}, Policy iterations: {len(planner.policy_history)}"
    )
    print(
        f"  Policy Iteration - Max Value: {np.max(pi_values):.3f}, Policy iterations: {len(planner_pi.policy_history)}"
    )
    print(f"  Pessimistic Planning - Max Value: {np.max(pessimistic_V):.3f}")
    print(f"  Optimistic Planning - Max Value: {np.max(optimistic_V):.3f}")

    policy_searcher = ModelBasedPolicySearch(
        tabular_model, num_states, num_actions, gamma=0.95
    )

    initial_state = env.start_state # Use environment's start state

    print("\n  Random Shooting (with Tabular Model):")
    best_sequence_rs, best_value_rs = policy_searcher.random_shooting(
        initial_state, horizon=5, num_sequences=500
    )
    print(
        f"    Best Value: {best_value_rs:.3f}, Best Sequence: {best_sequence_rs}"
    )

    print("\n  Cross-Entropy Method (with Tabular Model):")
    best_sequence_cem, best_value_cem = policy_searcher.cross_entropy_method(
        initial_state, horizon=5, num_sequences=200, num_elite=20, num_iterations=15
    )
    print(
        f"    Best Value: {best_value_cem:.3f}, Best Sequence: {best_sequence_cem}"
    )

    print("\n✅ Classical planning with learned models demonstration complete!")
    print("📊 Check 'visualizations/classical_planning.png' for plots.")
    print("""
Next: Dyna-Q algorithm - integrating planning and learning.
Explore `agents/dyna_q.py` for its implementation.
""")

if __name__ == "__main__":
    demonstrate_classical_planning()
