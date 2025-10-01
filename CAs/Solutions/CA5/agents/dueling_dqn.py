"""
Dueling DQN Implementation
=========================

This module implements Dueling DQN architecture which separates the
estimation of state value and state-dependent action advantages.

Key Features:
- Dueling DQN network architecture
- Advantage and value stream separation
- Performance analysis and visualization
- Comparison with standard DQN

Author: CA5 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .dqn_base import DQNAgent, device
import random
import warnings

warnings.filterwarnings("ignore")


class DuelingDQN(nn.Module):
    """Dueling DQN network with separate value and advantage streams"""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQN, self).__init__()

        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Value stream: estimates state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),  # Single value output
        )

        # Advantage stream: estimates action advantages A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),  # One advantage per action
        )

    def forward(self, x):
        """Forward pass with dueling combination"""
        # Shared features
        features = self.feature_layer(x)

        # Value and advantage streams
        value = self.value_stream(features)  # V(s)
        advantages = self.advantage_stream(features)  # A(s,a)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This ensures that the advantage function has zero mean
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class ConvDuelingDQN(nn.Module):
    """Convolutional Dueling DQN for visual inputs"""

    def __init__(self, input_shape, action_size):
        super(ConvDuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.action_size = action_size

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate flattened size
        conv_out_size = self._get_conv_out_size(input_shape)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, action_size)
        )

    def _get_conv_out_size(self, input_shape):
        """Calculate output size of convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self.conv_layers(dummy_input)
            return int(np.prod(conv_output.size()))

    def forward(self, x):
        """Forward pass"""
        # Convolutional features
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)

        # Value and advantage streams
        value = self.value_stream(conv_features)
        advantages = self.advantage_stream(conv_features)

        # Combine streams
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN agent with value-advantage architecture"""

    def __init__(self, state_size, action_size, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.agent_type = "Dueling DQN"

        # Override with dueling network
        if isinstance(state_size, tuple):  # Convolutional input
            self.q_network = ConvDuelingDQN(state_size, action_size).to(device)
        else:  # Fully connected input
            self.q_network = DuelingDQN(state_size, action_size).to(device)

        self.target_network = type(self.q_network)(state_size, action_size).to(device)
        self.update_target_network()

        # Reinitialize optimizer with new network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Additional tracking for analysis
        self.value_estimates = []
        self.advantage_estimates = []

    def get_value_advantage_estimates(self, state):
        """Extract value and advantage estimates for analysis"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get features
            if hasattr(self.q_network, "feature_layer"):
                features = self.q_network.feature_layer(state)
                value = self.q_network.value_stream(features).item()
                advantages = (
                    self.q_network.advantage_stream(features).squeeze(0).cpu().numpy()
                )
            else:
                # For convolutional networks, we can't easily extract intermediate values
                value = None
                advantages = None

        return value, advantages


class DuelingAnalysis:
    """Analyze Dueling DQN architecture and performance"""

    def __init__(self):
        self.results = {}

    def analyze_value_advantage_separation(self, agent, env, num_episodes=10):
        """Analyze how value and advantage streams behave"""
        print("Analyzing Value-Advantage Separation...")

        value_history = []
        advantage_history = []
        q_value_history = []

        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
            episode_values = []
            episode_advantages = []
            episode_q_values = []

            step_count = 0
            max_steps = 500

            while not done and step_count < max_steps:
                # Get estimates
                value, advantages = agent.get_value_advantage_estimates(state)

                if value is not None:
                    episode_values.append(value)
                    episode_advantages.append(advantages)
                    episode_q_values.append(value + (advantages - np.mean(advantages)))

                # Take action
                action = agent.get_action(state, training=False)
                result = env.step(action)
                next_state, reward, terminated, truncated, _ = result if len(result) == 5 else (*result, False)
                done = terminated or (truncated if len(result) == 5 else False)
                state = next_state
                step_count += 1

            value_history.append(episode_values)
            advantage_history.append(episode_advantages)
            q_value_history.append(episode_q_values)

        return {
            "value_history": value_history,
            "advantage_history": advantage_history,
            "q_value_history": q_value_history,
        }

    def visualize_dueling_architecture(self, agent, env):
        """Visualize the dueling architecture behavior"""
        analysis_data = self.analyze_value_advantage_separation(agent, env)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Value estimates over time
        if analysis_data["value_history"]:
            all_values = [
                v for episode in analysis_data["value_history"] for v in episode
            ]
            axes[0, 0].plot(all_values, alpha=0.7)
            axes[0, 0].set_title("State Value Estimates Over Time")
            axes[0, 0].set_xlabel("Time Step")
            axes[0, 0].set_ylabel("Value V(s)")
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Advantage distribution
        if analysis_data["advantage_history"]:
            all_advantages = np.array(
                [
                    adv
                    for episode in analysis_data["advantage_history"]
                    for adv in episode
                ]
            )
            if all_advantages.size > 0:
                for i in range(all_advantages.shape[1]):
                    axes[0, 1].hist(
                        all_advantages[:, i],
                        bins=20,
                        alpha=0.5,
                        label=f"Action {i}",
                        density=True,
                    )
                axes[0, 1].set_title("Advantage Distribution by Action")
                axes[0, 1].set_xlabel("Advantage A(s,a)")
                axes[0, 1].set_ylabel("Density")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-values vs Value + Advantages
        if analysis_data["q_value_history"] and analysis_data["value_history"]:
            # Take first episode for detailed analysis
            q_values = analysis_data["q_value_history"][0]
            values = analysis_data["value_history"][0]
            advantages = analysis_data["advantage_history"][0]

            if q_values and values and advantages:
                reconstructed_q = [
                    v + (adv - np.mean(adv)) for v, adv in zip(values, advantages)
                ]
                reconstructed_q = np.array(reconstructed_q).flatten()

                axes[0, 2].scatter(q_values, reconstructed_q, alpha=0.6, color="blue")
                axes[0, 2].plot(
                    [min(q_values), max(q_values)],
                    [min(q_values), max(q_values)],
                    "r--",
                    label="Perfect Reconstruction",
                )
                axes[0, 2].set_title("Q-Value Reconstruction")
                axes[0, 2].set_xlabel("Network Q(s,a)")
                axes[0, 2].set_ylabel("V(s) + (A(s,a) - mean(A))")
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)

        # 4. Value vs Advantage correlation
        if analysis_data["value_history"] and analysis_data["advantage_history"]:
            values_flat = [
                v for episode in analysis_data["value_history"] for v in episode
            ]
            advantages_flat = [
                adv for episode in analysis_data["advantage_history"] for adv in episode
            ]

            if values_flat and advantages_flat:
                # Correlation between value and max advantage
                max_advantages = [np.max(adv) for adv in advantages_flat]
                axes[1, 0].scatter(
                    values_flat, max_advantages, alpha=0.6, color="green"
                )
                axes[1, 0].set_title("Value vs Max Advantage Correlation")
                axes[1, 0].set_xlabel("State Value V(s)")
                axes[1, 0].set_ylabel("Max Advantage")
                axes[1, 0].grid(True, alpha=0.3)

        # 5. Advantage symmetry check
        if analysis_data["advantage_history"]:
            advantages_flat = [
                adv for episode in analysis_data["advantage_history"] for adv in episode
            ]
            if advantages_flat:
                mean_advantages = [np.mean(adv) for adv in advantages_flat]
                axes[1, 1].hist(mean_advantages, bins=20, alpha=0.7, color="purple")
                axes[1, 1].axvline(x=0, color="red", linestyle="--", label="Zero Mean")
                axes[1, 1].set_title("Advantage Mean Distribution")
                axes[1, 1].set_xlabel("Mean Advantage")
                axes[1, 1].set_ylabel("Frequency")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        # 6. Learning dynamics
        if hasattr(agent, "q_values") and hasattr(agent, "value_estimates"):
            if agent.value_estimates:
                axes[1, 2].plot(
                    agent.value_estimates, label="Value Estimates", alpha=0.7
                )
            if agent.q_values:
                axes[1, 2].plot(agent.q_values, label="Q-Value Estimates", alpha=0.7)
            axes[1, 2].set_title("Learning Dynamics")
            axes[1, 2].set_xlabel("Training Step")
            axes[1, 2].set_ylabel("Estimated Values")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_architectures(
        self, standard_agent, dueling_agent, env, num_episodes=200
    ):
        """Compare standard DQN vs Dueling DQN performance"""
        print("Comparing Standard DQN vs Dueling DQN...")

        # Train both agents
        print("Training Standard DQN...")
        standard_scores, _ = standard_agent.train(
            env, num_episodes, print_every=num_episodes // 4
        )

        print("Training Dueling DQN...")
        dueling_scores, _ = dueling_agent.train(
            env, num_episodes, print_every=num_episodes // 4
        )

        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        episodes = range(len(standard_scores))

        # 1. Learning curves
        axes[0, 0].plot(
            episodes, standard_scores, color="red", label="Standard DQN", linewidth=2
        )
        axes[0, 0].plot(
            episodes, dueling_scores, color="blue", label="Dueling DQN", linewidth=2
        )
        axes[0, 0].set_title("Learning Curves Comparison")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Rolling average performance
        window = 20
        standard_rolling = np.convolve(
            standard_scores, np.ones(window) / window, mode="valid"
        )
        dueling_rolling = np.convolve(
            dueling_scores, np.ones(window) / window, mode="valid"
        )

        axes[0, 1].plot(
            range(len(standard_rolling)),
            standard_rolling,
            color="red",
            label="Standard DQN",
            linewidth=2,
        )
        axes[0, 1].plot(
            range(len(dueling_rolling)),
            dueling_rolling,
            color="blue",
            label="Dueling DQN",
            linewidth=2,
        )
        axes[0, 1].set_title(f"Rolling Average Performance (Window={window})")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Average Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Final performance distribution
        final_window = 50
        standard_final = standard_scores[-final_window:]
        dueling_final = dueling_scores[-final_window:]

        axes[1, 0].boxplot(
            [standard_final, dueling_final], labels=["Standard DQN", "Dueling DQN"]
        )
        axes[1, 0].set_title(f"Final Performance (Last {final_window} Episodes)")
        axes[1, 0].set_ylabel("Episode Reward")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Performance improvement
        baseline_standard = np.mean(standard_scores[:50])
        baseline_dueling = np.mean(dueling_scores[:50])

        improvement_standard = [score - baseline_standard for score in standard_scores]
        improvement_dueling = [score - baseline_dueling for score in dueling_scores]

        axes[1, 1].plot(
            episodes,
            improvement_standard,
            color="red",
            label="Standard DQN",
            linewidth=2,
        )
        axes[1, 1].plot(
            episodes,
            improvement_dueling,
            color="blue",
            label="Dueling DQN",
            linewidth=2,
        )
        axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1, 1].set_title("Performance Improvement from Baseline")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Improvement")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\\nPerformance Comparison Summary:")
        print("=" * 40)
        print(
            f"Standard DQN Final Avg: {np.mean(standard_final):.2f} ± {np.std(standard_final):.2f}"
        )
        print(
            f"Dueling DQN Final Avg: {np.mean(dueling_final):.2f} ± {np.std(dueling_final):.2f}"
        )
        print(f"Improvement: {np.mean(dueling_final) - np.mean(standard_final):.2f}")

        return standard_scores, dueling_scores


# Example usage and demonstration
if __name__ == "__main__":
    print("Dueling DQN Implementation")
    print("=" * 40)

    # Test Dueling DQN agent creation
    agent = DuelingDQNAgent(state_size=4, action_size=2)
    print(f"Dueling DQN Agent created: {agent.agent_type}")

    # Test network architecture
    test_input = torch.randn(1, 4)
    output = agent.q_network(test_input)
    print(f"Network output shape: {output.shape}")

    # Test value-advantage extraction
    value, advantages = agent.get_value_advantage_estimates(
        test_input.squeeze(0).numpy()
    )
    if value is not None:
        print(f"Value estimate: {value:.4f}")
        print(f"Advantage estimates: {advantages}")

    print("\\n✓ Dueling DQN implementation complete")
    print("✓ Value-advantage architecture ready")
    print("✓ Analysis framework available")
