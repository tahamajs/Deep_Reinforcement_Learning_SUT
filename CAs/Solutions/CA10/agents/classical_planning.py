import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import TabularModel, NeuralModel, ModelTrainer, device
from environments.environments import SimpleGridWorld


class ModelBasedPlanner:
    """Classical planning algorithms using learned models"""

    def __init__(self, model, num_states, num_actions, gamma=0.99):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

        # Initialize value function and policy
        self.V = np.zeros(num_states)
        self.policy = np.zeros(num_states, dtype=int)

        # Planning history for analysis
        self.value_history = []
        self.policy_history = []

    def value_iteration(self, max_iterations=100, tolerance=1e-6):
        """Value Iteration using learned model"""

        print(f"Running Value Iteration (max_iter={max_iterations}, tol={tolerance})")

        for iteration in range(max_iterations):
            old_V = self.V.copy()

            for state in range(self.num_states):
                # Compute Q-values for all actions
                q_values = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    # Compute expected value using model
                    expected_value = 0

                    for next_state in range(self.num_states):
                        transition_prob = self.model.get_transition_prob(
                            state, action, next_state
                        )
                        reward = self.model.get_expected_reward(state, action)
                        expected_value += transition_prob * (
                            reward + self.gamma * old_V[next_state]
                        )

                    q_values[action] = expected_value

                # Update value and policy
                self.V[state] = np.max(q_values)
                self.policy[state] = np.argmax(q_values)

            # Store history
            self.value_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())

            # Check convergence
            if np.max(np.abs(self.V - old_V)) < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

        return self.V, self.policy

    def policy_iteration(self, max_iterations=50, eval_max_iterations=100):
        """Policy Iteration using learned model"""

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
                q_values = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    expected_value = 0

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

            # Store history
            self.value_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())

            # Check convergence
            if np.array_equal(self.policy, old_policy):
                print(f"Converged after {iteration + 1} iterations")
                break

        return self.V, self.policy

    def policy_evaluation(self, policy, max_iterations=100, tolerance=1e-6):
        """Evaluate a given policy using learned model"""

        V = np.zeros(self.num_states)

        for iteration in range(max_iterations):
            old_V = V.copy()

            for state in range(self.num_states):
                action = policy[state]
                expected_value = 0

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

    def compute_q_function(self):
        """Compute Q-function from current value function"""

        Q = np.zeros((self.num_states, self.num_actions))

        for state in range(self.num_states):
            for action in range(self.num_actions):
                expected_value = 0

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
    """Planning with model uncertainty"""

    def __init__(self, ensemble_model, num_states, num_actions, gamma=0.99):
        self.ensemble_model = ensemble_model
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

    def pessimistic_value_iteration(self, beta=1.0, max_iterations=100):
        """Value iteration with pessimistic model estimates"""

        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states, dtype=int)

        print(f"Running Pessimistic Value Iteration (beta={beta})")

        # Set model to eval mode to avoid gradient computation
        self.ensemble_model.eval()

        for iteration in range(max_iterations):
            old_V = V.copy()

            for state in range(self.num_states):
                q_values = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    # Get predictions from ensemble
                    state_onehot = np.eye(self.num_states)[state : state + 1]
                    action_tensor = np.array([action])

                    # Convert to tensors
                    state_tensor = torch.FloatTensor(state_onehot).to(device)
                    action_tensor = torch.LongTensor(action_tensor).to(device)

                    # Get ensemble predictions
                    next_state_mean, reward_mean, next_state_std, reward_std = (
                        self.ensemble_model.predict_with_uncertainty(
                            state_tensor, action_tensor
                        )
                    )

                    # Use pessimistic estimates
                    pessimistic_reward = (
                        reward_mean.cpu().item() - beta * reward_std.cpu().item()
                    )

                    # For simplicity, assume deterministic next state (can be extended)
                    next_state_pred = next_state_mean.cpu().detach().numpy()[0]
                    next_state_idx = np.argmax(
                        next_state_pred
                    )  # Most likely next state

                    q_values[action] = (
                        pessimistic_reward + self.gamma * old_V[next_state_idx]
                    )

                V[state] = np.max(q_values)
                policy[state] = np.argmax(q_values)

            if np.max(np.abs(V - old_V)) < 1e-6:
                print(f"Converged after {iteration + 1} iterations")
                break

        return V, policy

    def optimistic_value_iteration(self, beta=1.0, max_iterations=100):
        """Value iteration with optimistic model estimates"""

        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states, dtype=int)

        print(f"Running Optimistic Value Iteration (beta={beta})")

        # Set model to eval mode to avoid gradient computation
        self.ensemble_model.eval()

        for iteration in range(max_iterations):
            old_V = V.copy()

            for state in range(self.num_states):
                q_values = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    # Get predictions from ensemble
                    state_onehot = np.eye(self.num_states)[state : state + 1]
                    action_tensor = np.array([action])

                    # Convert to tensors
                    state_tensor = torch.FloatTensor(state_onehot).to(device)
                    action_tensor = torch.LongTensor(action_tensor).to(device)

                    # Get ensemble predictions
                    next_state_mean, reward_mean, next_state_std, reward_std = (
                        self.ensemble_model.predict_with_uncertainty(
                            state_tensor, action_tensor
                        )
                    )

                    # Use optimistic estimates
                    optimistic_reward = (
                        reward_mean.cpu().item() + beta * reward_std.cpu().item()
                    )

                    # For simplicity, assume deterministic next state
                    next_state_pred = next_state_mean.cpu().detach().numpy()[0]
                    next_state_idx = np.argmax(next_state_pred)

                    q_values[action] = (
                        optimistic_reward + self.gamma * old_V[next_state_idx]
                    )

                V[state] = np.max(q_values)
                policy[state] = np.argmax(q_values)

            if np.max(np.abs(V - old_V)) < 1e-6:
                print(f"Converged after {iteration + 1} iterations")
                break

        return V, policy


class ModelBasedPolicySearch:
    """Policy search using learned models"""

    def __init__(self, model, state_dim, action_dim, gamma=0.99):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

    def random_shooting(self, initial_state, horizon=10, num_sequences=1000):
        """Random shooting with learned model"""
        best_sequence = None
        best_value = -np.inf

        for _ in range(num_sequences):
            # Sample random action sequence
            action_sequence = np.random.randint(0, self.action_dim, horizon)

            # Evaluate sequence using model
            total_reward = 0
            current_state = initial_state
            discount = 1.0

            for action in action_sequence:
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
        initial_state,
        horizon=10,
        num_sequences=1000,
        num_elite=100,
        num_iterations=10,
    ):
        """Cross-entropy method for policy search"""

        # Initialize action probabilities (uniform)
        action_probs = np.ones((horizon, self.action_dim)) / self.action_dim

        for iteration in range(num_iterations):
            # Sample action sequences
            sequences = []
            values = []

            for _ in range(num_sequences):
                sequence = []
                for t in range(horizon):
                    action = np.random.choice(self.action_dim, p=action_probs[t])
                    sequence.append(action)

                # Evaluate sequence
                total_reward = 0
                current_state = initial_state
                discount = 1.0

                for action in sequence:
                    next_state, reward = self.model.sample_transition(
                        current_state, action
                    )
                    total_reward += discount * reward
                    discount *= self.gamma
                    current_state = next_state

                sequences.append(sequence)
                values.append(total_reward)

            # Select elite sequences
            elite_indices = np.argsort(values)[-num_elite:]
            elite_sequences = [sequences[i] for i in elite_indices]

            # Update action probabilities
            action_counts = np.zeros((horizon, self.action_dim))

            for sequence in elite_sequences:
                for t, action in enumerate(sequence):
                    action_counts[t, action] += 1

            # Smooth update
            alpha = 0.7
            new_probs = action_counts / num_elite
            action_probs = alpha * new_probs + (1 - alpha) * action_probs

            # Add small amount of noise for exploration
            action_probs += 0.01 / self.action_dim
            action_probs /= np.sum(action_probs, axis=1, keepdims=True)

        # Return best sequence
        best_sequence = elite_sequences[np.argmax([values[i] for i in elite_indices])]
        best_value = max([values[i] for i in elite_indices])

        return best_sequence, best_value


def demonstrate_classical_planning():
    """Demonstrate classical planning with learned models"""

    print("Classical Planning with Learned Models")
    print("=" * 50)

    # Create environment and collect data
    env = SimpleGridWorld(size=4)
    tabular_model = TabularModel(env.num_states, env.num_actions)

    # Collect experience
    n_episodes = 200
    experience_data = []

    print("\n1. Collecting experience for model learning...")
    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = np.random.randint(env.num_actions)  # Random policy
            next_state, reward, done = env.step(action)

            # Update tabular model
            tabular_model.update(state, action, next_state, reward)

            # Store for neural model
            experience_data.append((state, action, next_state, reward))

            state = next_state

    print(f"Collected {len(experience_data)} transitions")

    # Prepare data for neural model
    states = np.array([exp[0] for exp in experience_data])
    actions = np.array([exp[1] for exp in experience_data])
    next_states = np.array([exp[2] for exp in experience_data])
    rewards = np.array([exp[3] for exp in experience_data])

    # Convert states to one-hot for neural model
    states_onehot = np.eye(env.num_states)[states]
    next_states_onehot = np.eye(env.num_states)[next_states]

    # Train neural model
    print("\n2. Training neural model...")
    neural_model = NeuralModel(
        env.num_states, env.num_actions, hidden_dim=64, ensemble_size=1
    ).to(device)
    trainer = ModelTrainer(neural_model, lr=1e-3)

    trainer.train_batch(
        (states_onehot, actions, next_states_onehot, rewards), epochs=15, batch_size=64
    )

    # Use the tabular model we learned earlier
    planner = ModelBasedPlanner(
        tabular_model, env.num_states, env.num_actions, gamma=0.95
    )

    print("\n3. Value Iteration with Learned Model:")
    vi_values, vi_policy = planner.value_iteration(max_iterations=50)

    print("\n4. Policy Iteration with Learned Model:")
    planner_pi = ModelBasedPlanner(
        tabular_model, env.num_states, env.num_actions, gamma=0.95
    )
    pi_values, pi_policy = planner_pi.policy_iteration(max_iterations=20)

    # Compare with uncertainty-aware planning
    print("\n5. Uncertainty-Aware Planning:")
    uncertainty_planner = UncertaintyAwarePlanner(
        neural_model, env.num_states, env.num_actions
    )
    pessimistic_V, pessimistic_policy = uncertainty_planner.pessimistic_value_iteration(
        beta=0.5
    )
    optimistic_V, optimistic_policy = uncertainty_planner.optimistic_value_iteration(
        beta=0.5
    )

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Reshape values and policies for visualization
    grid_size = int(np.sqrt(env.num_states))

    def plot_value_function(ax, values, title):
        value_grid = values.reshape(grid_size, grid_size)
        im = ax.imshow(value_grid, cmap="viridis")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    def plot_policy(ax, policy, title):
        policy_grid = policy.reshape(grid_size, grid_size)
        # Map actions to arrows: 0=↑, 1=↓, 2=←, 3=→
        arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

        ax.imshow(np.zeros((grid_size, grid_size)), cmap="gray", alpha=0.3)

        for i in range(grid_size):
            for j in range(grid_size):
                action = policy_grid[i, j]
                ax.text(
                    j,
                    i,
                    arrow_map[action],
                    ha="center",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    color="blue",
                )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot value functions
    plot_value_function(axes[0, 0], vi_values, "Value Iteration - Values")
    plot_value_function(axes[0, 1], pi_values, "Policy Iteration - Values")
    plot_value_function(axes[0, 2], pessimistic_V, "Pessimistic Planning - Values")

    # Plot policies
    plot_policy(axes[1, 0], vi_policy, "Value Iteration - Policy")
    plot_policy(axes[1, 1], pi_policy, "Policy Iteration - Policy")
    plot_policy(axes[1, 2], pessimistic_policy, "Pessimistic Planning - Policy")

    plt.tight_layout()
    plt.savefig("visualizations/classical_planning.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Compare planning methods
    print("\n6. Planning Method Comparison:")
    print(
        f"Value Iteration - Max Value: {np.max(vi_values):.3f}, Policy Changes: {len(planner.value_history)}"
    )
    print(
        f"Policy Iteration - Max Value: {np.max(pi_values):.3f}, Policy Changes: {len(planner_pi.value_history)}"
    )
    print(f"Pessimistic Planning - Max Value: {np.max(pessimistic_V):.3f}")
    print(f"Optimistic Planning - Max Value: {np.max(optimistic_V):.3f}")

    # Test policy search methods
    print("\n7. Model-Based Policy Search:")
    policy_searcher = ModelBasedPolicySearch(
        tabular_model, env.num_states, env.num_actions
    )

    # Random shooting
    initial_state = 0
    best_sequence_rs, best_value_rs = policy_searcher.random_shooting(
        initial_state, horizon=5, num_sequences=500
    )
    print(
        f"Random Shooting - Best Value: {best_value_rs:.3f}, Best Sequence: {best_sequence_rs}"
    )

    # Cross-entropy method
    best_sequence_cem, best_value_cem = policy_searcher.cross_entropy_method(
        initial_state, horizon=5, num_sequences=200, num_elite=20
    )
    print(
        f"Cross-Entropy Method - Best Value: {best_value_cem:.3f}, Best Sequence: {best_sequence_cem}"
    )

    print("\n✅ Classical planning with learned models complete!")
    print("📊 Next: Dyna-Q algorithm - integrating planning and learning")
