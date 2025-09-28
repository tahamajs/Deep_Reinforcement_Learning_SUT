"""
Rainbow DQN Implementation
=========================

This module implements Rainbow DQN which combines multiple DQN improvements:
- Double DQN (addresses overestimation)
- Dueling DQN (separates value and advantage)
- Prioritized Experience Replay (improves sample efficiency)
- Multi-step learning (bootstrapped n-step returns)
- Distributional RL (learns value distributions)
- Noisy nets (exploration through noise)

Key Features:
- RainbowDQNAgent combining all improvements
- Distributional value learning
- Multi-step TD learning
- Noisy linear layers for exploration
- Comprehensive analysis and comparison tools

Author: CA5 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dqn_base import DQNAgent, device
from double_dqn import DoubleDQNAgent
from dueling_dqn import DuelingDQN
from prioritized_replay import PrioritizedReplayBuffer
import random
import math
import warnings

warnings.filterwarnings("ignore")


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""

    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # Register buffers for noise
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def reset_noise(self):
        """Reset noise parameters"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Scale noise for stability"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        """Forward pass with noise"""
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class RainbowDQN(nn.Module):
    """Rainbow DQN network combining all improvements"""

    def __init__(self, state_size, action_size, n_atoms=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()

        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Distributional atoms
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)

        # Feature layer with noisy nets
        self.feature_layer = nn.Sequential(NoisyLinear(state_size, 128), nn.ReLU())

        # Value stream (dueling architecture)
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, n_atoms),  # Distribution over values
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_size * n_atoms),  # Distribution per action
        )

    def forward(self, x):
        """Forward pass returning value distributions"""
        batch_size = x.size(0)

        # Shared features
        features = self.feature_layer(x)

        # Value distribution: V(x) -> P(V|x)
        value_dist = self.value_stream(features).view(batch_size, 1, self.n_atoms)

        # Advantage distributions: A(x,a) -> P(A|x,a) for each action
        advantage_dist = self.advantage_stream(features).view(
            batch_size, self.action_size, self.n_atoms
        )

        # Combine: Q(x,a) = V(x) + (A(x,a) - mean(A(x,a')))
        q_dist = value_dist + advantage_dist - advantage_dist.mean(dim=1, keepdim=True)

        # Apply softmax to get proper distributions
        q_dist = F.softmax(q_dist, dim=2)

        return q_dist

    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowDQNAgent(DQNAgent):
    """Rainbow DQN agent combining all DQN improvements"""

    def __init__(
        self,
        state_size,
        action_size,
        n_step=3,
        n_atoms=51,
        v_min=-10,
        v_max=10,
        **kwargs,
    ):
        super().__init__(state_size, action_size, **kwargs)
        self.agent_type = "Rainbow DQN"

        # Rainbow parameters
        self.n_step = n_step
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)

        # Override network with Rainbow architecture
        self.q_network = RainbowDQN(state_size, action_size, n_atoms, v_min, v_max).to(
            device
        )
        self.target_network = RainbowDQN(
            state_size, action_size, n_atoms, v_min, v_max
        ).to(device)
        self.update_target_network()

        # Override memory with prioritized buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=10000,
            alpha=0.5,  # Lower alpha for stability
            beta=0.4,
            beta_increment=1e-6,
        )

        # Multi-step buffer for n-step returns
        self.n_step_buffer = []
        self.gamma_n = self.gamma**n_step

        # Reinitialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Additional tracking
        self.distributions = []

    def act(self, state, epsilon=None):
        """Rainbow action selection with noise reset"""
        # Reset noise for exploration
        self.q_network.reset_noise()

        return super().act(state, epsilon)

    def get_q_values(self, state):
        """Get Q-value expectations from distributions"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)

            dist = self.q_network(state)
            # Expected Q-values: sum over atoms
            q_values = torch.sum(dist * self.support, dim=2)

        return q_values.squeeze(0).cpu().numpy()

    def get_value_distribution(self, state, action=None):
        """Get value distribution for analysis"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(device)

            dist = self.q_network(state)

            if action is not None:
                return dist[0, action].cpu().numpy()
            else:
                return dist[0].cpu().numpy()

    def store_n_step_experience(self, experience):
        """Store experience in n-step buffer"""
        self.n_step_buffer.append(experience)

        if len(self.n_step_buffer) >= self.n_step:
            # Create n-step experience
            n_step_experience = self._get_n_step_info()
            self.memory.add(n_step_experience)
            self.n_step_buffer.pop(0)  # Remove oldest experience

    def _get_n_step_info(self):
        """Calculate n-step return"""
        reward, next_state, done = (
            0,
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4],
        )

        # Calculate n-step discounted reward
        for i in range(self.n_step):
            reward += (self.gamma**i) * self.n_step_buffer[i][2]

        # Original state and action
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]

        return (state, action, reward, next_state, done)

    def train_step(self):
        """Rainbow DQN training step with distributional RL"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch with priorities
        experiences, idxs, is_weights = self.memory.sample(self.batch_size)
        batch = self.experience_to_batch(experiences)

        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(device)

        # Reset noise for training
        self.q_network.reset_noise()

        # Current distributions
        current_dist = self.q_network(states)
        current_action_dist = current_dist[range(self.batch_size), actions.squeeze()]

        # Target distributions (Double Q-learning + distributional)
        with torch.no_grad():
            # Action selection: online network
            next_q_values = self.get_q_values(next_states)
            next_actions = np.argmax(next_q_values, axis=1)

            # Action evaluation: target network
            next_dist = self.target_network(next_states)
            next_action_dist = next_dist[range(self.batch_size), next_actions]

            # Project distributional target
            target_dist = self._project_distribution(next_action_dist, rewards, dones)

        # KL divergence loss (cross-entropy between distributions)
        loss = -(target_dist * torch.log(current_action_dist + 1e-8)).sum(dim=1).mean()

        # Weighted loss for prioritized replay
        loss = (is_weights * loss.unsqueeze(1)).mean()

        # Update priorities with TD errors (approximated)
        td_errors = (
            torch.sum(torch.abs(current_action_dist - target_dist), dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        self.memory.update_priorities(idxs, td_errors)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        # Decay epsilon (though Rainbow uses noise for exploration)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Store training metrics
        self.losses.append(loss.item())

        # Store distribution for analysis
        self.distributions.append(current_dist[0].detach().cpu().numpy())

        return loss.item()

    def _project_distribution(self, next_dist, rewards, dones):
        """Project distributional target for n-step returns"""
        batch_size = rewards.size(0)

        # Create target distribution
        target_dist = torch.zeros(batch_size, self.n_atoms).to(device)

        for i in range(batch_size):
            if dones[i]:
                # Terminal state: reward only
                target_value = rewards[i].item()
            else:
                # Non-terminal: reward + gamma^n * expected value
                expected_next_value = torch.sum(next_dist[i] * self.support)
                target_value = (
                    rewards[i].item() + self.gamma_n * expected_next_value.item()
                )

            # Project target value onto support
            target_dist[i] = self._categorical_projection(target_value)

        return target_dist

    def _categorical_projection(self, value):
        """Project a scalar value onto categorical distribution"""
        # Find atoms above and below the value
        below = torch.where(self.support <= value)[0]
        above = torch.where(self.support > value)[0]

        if len(below) == 0:
            # Value is below all atoms
            dist = torch.zeros(self.n_atoms).to(device)
            dist[0] = 1.0
            return dist
        elif len(above) == 0:
            # Value is above all atoms
            dist = torch.zeros(self.n_atoms).to(device)
            dist[-1] = 1.0
            return dist
        else:
            # Interpolate between atoms
            idx_below = below[-1]
            idx_above = above[0]

            atom_below = self.support[idx_below]
            atom_above = self.support[idx_above]

            # Linear interpolation
            if atom_above == atom_below:
                prob = 1.0
            else:
                prob = (atom_above - value) / (atom_above - atom_below)

            dist = torch.zeros(self.n_atoms).to(device)
            dist[idx_below] = prob
            dist[idx_above] = 1.0 - prob

            return dist


class RainbowAnalysis:
    """Analyze Rainbow DQN behavior and components"""

    def __init__(self):
        self.results = {}

    def analyze_value_distributions(self, agent, env, num_episodes=5):
        """Analyze learned value distributions"""
        print("Analyzing Rainbow DQN Value Distributions...")

        distributions_over_time = []
        q_values_over_time = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_distributions = []
            episode_q_values = []

            while not done:
                # Get value distribution for current state
                dist = agent.get_value_distribution(state)
                q_values = agent.get_q_values(state)

                episode_distributions.append(dist)
                episode_q_values.append(q_values)

                # Take action
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state

            distributions_over_time.append(episode_distributions)
            q_values_over_time.append(episode_q_values)

        return {
            "distributions": distributions_over_time,
            "q_values": q_values_over_time,
        }

    def visualize_rainbow_components(self, agent, env):
        """Visualize Rainbow DQN components"""
        analysis_data = self.analyze_value_distributions(agent, env)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Value distribution evolution
        if analysis_data["distributions"]:
            first_episode_distributions = analysis_data["distributions"][0]
            support = agent.support.cpu().numpy()

            for i, dist in enumerate(
                first_episode_distributions[:10]
            ):  # First 10 steps
                axes[0, 0].plot(support, dist, alpha=0.6, label=f"Step {i+1}")

            axes[0, 0].set_title("Value Distribution Evolution")
            axes[0, 0].set_xlabel("Value")
            axes[0, 0].set_ylabel("Probability")
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Q-value expectations
        if analysis_data["q_values"]:
            first_episode_q_values = analysis_data["q_values"][0]
            q_values_array = np.array(first_episode_q_values)

            for action in range(q_values_array.shape[1]):
                axes[0, 1].plot(
                    q_values_array[:, action], label=f"Action {action}", linewidth=2
                )

            axes[0, 1].set_title("Q-Value Expectations Over Time")
            axes[0, 1].set_xlabel("Time Step")
            axes[0, 1].set_ylabel("Q-Value")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution variance
        if analysis_data["distributions"]:
            variances = []
            for episode_dists in analysis_data["distributions"]:
                episode_variances = [np.var(dist) for dist in episode_dists]
                variances.extend(episode_variances)

            axes[0, 2].plot(variances, color="purple", linewidth=2)
            axes[0, 2].set_title("Value Distribution Variance")
            axes[0, 2].set_xlabel("Time Step")
            axes[0, 2].set_ylabel("Variance")
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Action selection confidence
        if analysis_data["q_values"]:
            confidences = []
            for episode_q in analysis_data["q_values"]:
                for q_values in episode_q:
                    # Confidence as max Q minus second max Q
                    sorted_q = np.sort(q_values)
                    confidence = sorted_q[-1] - sorted_q[-2] if len(sorted_q) > 1 else 0
                    confidences.append(confidence)

            axes[1, 0].plot(confidences, color="green", linewidth=2)
            axes[1, 0].set_title("Action Selection Confidence")
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_ylabel("Confidence")
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Distribution entropy
        if analysis_data["distributions"]:
            entropies = []
            for episode_dists in analysis_data["distributions"]:
                for dist in episode_dists:
                    # Calculate entropy: -sum(p * log p)
                    entropy = -np.sum(dist * np.log(dist + 1e-8))
                    entropies.append(entropy)

            axes[1, 1].plot(entropies, color="orange", linewidth=2)
            axes[1, 1].set_title("Value Distribution Entropy")
            axes[1, 1].set_xlabel("Time Step")
            axes[1, 1].set_ylabel("Entropy")
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Component comparison (placeholder for multi-agent comparison)
        axes[1, 2].text(
            0.5,
            0.5,
            "Rainbow DQN combines:\\n- Double Q-learning\\n- Dueling architecture\\n- Prioritized replay\\n- Distributional RL\\n- Noisy exploration\\n- Multi-step learning",
            transform=axes[1, 2].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        )
        axes[1, 2].set_title("Rainbow Components")
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def compare_rainbow_vs_components(
        self, standard_agent, rainbow_agent, env, num_episodes=200
    ):
        """Compare Rainbow DQN vs individual components"""
        print("Comparing Rainbow DQN vs Standard DQN...")

        # Train both agents
        print("Training Standard DQN...")
        standard_scores, _ = standard_agent.train(
            env, num_episodes, print_every=num_episodes // 4
        )

        print("Training Rainbow DQN...")
        rainbow_scores, _ = rainbow_agent.train(
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
            episodes, rainbow_scores, color="purple", label="Rainbow DQN", linewidth=2
        )
        axes[0, 0].set_title("Rainbow vs Standard DQN Learning Curves")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Performance improvement
        baseline_standard = np.mean(standard_scores[:30])
        baseline_rainbow = np.mean(rainbow_scores[:30])

        improvement_standard = [score - baseline_standard for score in standard_scores]
        improvement_rainbow = [score - baseline_rainbow for score in rainbow_scores]

        axes[0, 1].plot(
            episodes,
            improvement_standard,
            color="red",
            label="Standard DQN",
            linewidth=2,
        )
        axes[0, 1].plot(
            episodes,
            improvement_rainbow,
            color="purple",
            label="Rainbow DQN",
            linewidth=2,
        )
        axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 1].set_title("Performance Improvement from Baseline")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Improvement")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Final performance
        final_window = 50
        standard_final = standard_scores[-final_window:]
        rainbow_final = rainbow_scores[-final_window:]

        axes[1, 0].boxplot(
            [standard_final, rainbow_final], labels=["Standard DQN", "Rainbow DQN"]
        )
        axes[1, 0].set_title(f"Final Performance (Last {final_window} Episodes)")
        axes[1, 0].set_ylabel("Episode Reward")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Stability analysis
        stability_standard = np.std(standard_scores[-100:])
        stability_rainbow = np.std(rainbow_scores[-100:])

        axes[1, 1].bar(
            ["Standard DQN", "Rainbow DQN"],
            [stability_standard, stability_rainbow],
            color=["red", "purple"],
            alpha=0.7,
        )
        axes[1, 1].set_title("Performance Stability (Std Dev)")
        axes[1, 1].set_ylabel("Standard Deviation")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\\nRainbow vs Standard DQN Summary:")
        print("=" * 40)
        print(
            f"Standard DQN Final Avg: {np.mean(standard_final):.2f} ± {np.std(standard_final):.2f}"
        )
        print(
            f"Rainbow DQN Final Avg: {np.mean(rainbow_final):.2f} ± {np.std(rainbow_final):.2f}"
        )
        print(f"Improvement: {np.mean(rainbow_final) - np.mean(standard_final):.2f}")
        print(f"Stability Improvement: {stability_standard - stability_rainbow:.2f}")

        return standard_scores, rainbow_scores


# Example usage and demonstration
if __name__ == "__main__":
    print("Rainbow DQN Implementation")
    print("=" * 30)

    # Test Rainbow DQN agent creation
    agent = RainbowDQNAgent(state_size=4, action_size=2)
    print(f"Rainbow DQN Agent created: {agent.agent_type}")

    # Test network components
    test_input = torch.randn(1, 4)
    output_dist = agent.q_network(test_input)
    print(f"Output distribution shape: {output_dist.shape}")

    q_values = agent.get_q_values(test_input.squeeze(0).numpy())
    print(f"Q-values: {q_values}")

    distribution = agent.get_value_distribution(test_input.squeeze(0).numpy())
    print(f"Value distribution shape: {distribution.shape}")

    print("\\n✓ Rainbow DQN implementation complete")
    print("✓ Distributional RL ready")
    print("✓ Multi-step learning configured")
    print("✓ Noisy exploration enabled")
