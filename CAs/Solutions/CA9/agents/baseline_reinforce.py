import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from ..utils.utils import device


class ValueNetwork(nn.Module):
    """Value network for baseline estimation"""

    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze()


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)

    def get_action_and_log_prob(self, state):
        """Get action and its log probability"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        action_probs = self.forward(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob


class BaselineREINFORCEAgent:
    """REINFORCE with baseline for variance reduction"""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        baseline_type="value_function",
        hidden_dim=128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.baseline_type = baseline_type

        # Policy network
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Value network for baseline
        if baseline_type == "value_function":
            self.value_network = ValueNetwork(state_dim, hidden_dim).to(device)
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []

        # Training history
        self.episode_rewards_history = []
        self.policy_losses = []
        self.value_losses = []
        self.advantage_variances = []

        # Moving average baseline
        if baseline_type == "moving_average":
            self.moving_avg_baseline = 0.0
            self.baseline_alpha = 0.01

    def select_action(self, state):
        """Select action based on current policy"""
        action, log_prob = self.policy_network.get_action_and_log_prob(state)

        # Get value estimate if using value function baseline
        if self.baseline_type == "value_function":
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            else:
                state_tensor = state
            with torch.no_grad():
                value = self.value_network(state_tensor).item()
        else:
            value = 0.0

        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_values.append(value)

        return action

    def store_reward(self, reward):
        """Store reward for current episode"""
        self.episode_rewards.append(reward)

    def calculate_returns_and_advantages(self):
        """Calculate returns and advantages with baseline"""
        returns = []
        advantages = []

        # Calculate discounted returns
        discounted_sum = 0
        for reward in reversed(self.episode_rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.FloatTensor(returns).to(device)

        if self.baseline_type == "value_function":
            # Use value function as baseline
            values = torch.FloatTensor(self.episode_values).to(device)
            advantages = returns - values
        elif self.baseline_type == "moving_average":
            # Use moving average as baseline
            self.moving_avg_baseline = (
                1 - self.baseline_alpha
            ) * self.moving_avg_baseline + self.baseline_alpha * returns.mean().item()
            advantages = returns - self.moving_avg_baseline
        else:
            # No baseline
            advantages = returns

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update_policy(self):
        """Update policy using REINFORCE with baseline"""
        if len(self.episode_log_probs) == 0:
            return

        returns, advantages = self.calculate_returns_and_advantages()

        # Calculate policy loss
        policy_loss = []
        for log_prob, advantage in zip(self.episode_log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update value function if using value function baseline
        if self.baseline_type == "value_function":
            states_tensor = torch.FloatTensor(self.episode_states).to(device)
            values_pred = self.value_network(states_tensor)
            value_loss = F.mse_loss(values_pred, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(), max_norm=1.0
            )
            self.value_optimizer.step()

            self.value_losses.append(value_loss.item())

        # Store metrics
        self.policy_losses.append(policy_loss.item())
        self.advantage_variances.append(advantages.var().item())

        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        self.episode_values.clear()

    def train_episode(self, env, max_steps=1000):
        """Train for one episode"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.store_reward(reward)
            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.update_policy()
        self.episode_rewards_history.append(total_reward)

        return total_reward, steps

    def evaluate(self, env, num_episodes=10):
        """Evaluate current policy"""
        self.policy_network.eval()
        if self.baseline_type == "value_function":
            self.value_network.eval()

        rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0

            for _ in range(1000):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_probs = self.policy_network(state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if done:
                    break

                state = next_state

            rewards.append(total_reward)

        self.policy_network.train()
        if self.baseline_type == "value_function":
            self.value_network.train()

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }


class VarianceAnalyzer:
    """Analyze variance reduction techniques"""

    def compare_baseline_methods(self, env_name="CartPole-v1", num_episodes=200):
        """Compare different baseline methods"""

        print("=" * 70)
        print("Variance Reduction Techniques Comparison")
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        baseline_types = ["none", "moving_average", "value_function"]
        results = {}

        for baseline_type in baseline_types:
            print(f"\nTraining {baseline_type.replace('_', ' ').title()}...")

            agent = BaselineREINFORCEAgent(
                state_dim, action_dim, baseline_type=baseline_type, lr=1e-3
            )

            for episode in range(num_episodes):
                reward, steps = agent.train_episode(env)

                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(agent.episode_rewards_history[-20:])
                    print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f}")

            eval_results = agent.evaluate(env, 20)

            results[baseline_type] = {
                "agent": agent,
                "eval_results": eval_results,
                "final_performance": np.mean(agent.episode_rewards_history[-20:]),
            }

        env.close()

        self._visualize_baseline_comparison(results)

        return results

    def _visualize_baseline_comparison(self, results):
        """Visualize baseline comparison results"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Learning curves
        ax = axes[0, 0]
        colors = ["red", "orange", "green"]

        for i, (baseline_type, data) in enumerate(results.items()):
            agent = data["agent"]
            rewards = agent.episode_rewards_history

            if len(rewards) > 10:
                smoothed = pd.Series(rewards).rolling(window=20).mean()
                ax.plot(
                    smoothed,
                    label=baseline_type.replace("_", " ").title(),
                    color=colors[i],
                    linewidth=2,
                )

        ax.set_title("Learning Curves Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Final performance comparison
        ax = axes[0, 1]
        baseline_names = [bt.replace("_", " ").title() for bt in results.keys()]
        final_performances = [data["final_performance"] for data in results.values()]
        eval_means = [data["eval_results"]["mean_reward"] for data in results.values()]
        eval_stds = [data["eval_results"]["std_reward"] for data in results.values()]

        x = np.arange(len(baseline_names))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            final_performances,
            width,
            label="Training",
            alpha=0.7,
            color=colors,
        )
        bars2 = ax.bar(
            x + width / 2,
            eval_means,
            width,
            yerr=eval_stds,
            label="Evaluation",
            alpha=0.7,
        )

        ax.set_xlabel("Baseline Method")
        ax.set_ylabel("Average Reward")
        ax.set_title("Final Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(baseline_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Advantage variance
        ax = axes[1, 0]
        for i, (baseline_type, data) in enumerate(results.items()):
            agent = data["agent"]
            if agent.advantage_variances:
                variances = agent.advantage_variances
                if len(variances) > 20:
                    smoothed = pd.Series(variances).rolling(window=20).mean()
                    ax.plot(
                        smoothed,
                        label=baseline_type.replace("_", " ").title(),
                        color=colors[i],
                        linewidth=2,
                    )

        ax.set_title("Advantage Variance Evolution")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Advantage Variance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Policy loss comparison
        ax = axes[1, 1]
        for i, (baseline_type, data) in enumerate(results.items()):
            agent = data["agent"]
            if agent.policy_losses:
                losses = agent.policy_losses
                if len(losses) > 20:
                    smoothed = pd.Series(losses).rolling(window=20).mean()
                    ax.plot(
                        smoothed,
                        label=baseline_type.replace("_", " ").title(),
                        color=colors[i],
                        linewidth=2,
                    )

        ax.set_title("Policy Loss Evolution")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Policy Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 50)
        print("VARIANCE REDUCTION SUMMARY")
        print("=" * 50)

        for baseline_type, data in results.items():
            final_perf = data["final_performance"]
            eval_perf = data["eval_results"]["mean_reward"]
            eval_std = data["eval_results"]["std_reward"]

            print(f"\n{baseline_type.replace('_', ' ').title()}:")
            print(f"  Final Training Performance: {final_perf:.2f}")
            print(f"  Evaluation Performance: {eval_perf:.2f} ± {eval_std:.2f}")

            agent = data["agent"]
            if agent.advantage_variances:
                avg_variance = np.mean(agent.advantage_variances[-50:])
                print(f"  Average Advantage Variance (last 50): {avg_variance:.4f}")


# Example usage
if __name__ == "__main__":
    analyzer = VarianceAnalyzer()
    results = analyzer.compare_baseline_methods("CartPole-v1", num_episodes=200)
