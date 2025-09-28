import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from reinforce import REINFORCEAgent
from utils import device


class BaselineREINFORCEAgent(REINFORCEAgent):
    """REINFORCE with baseline for variance reduction"""

    def __init__(
        self, state_dim, action_dim, lr=1e-3, gamma=0.99, baseline_type="moving_average"
    ):
        super().__init__(state_dim, action_dim, lr, gamma)

        self.baseline_type = baseline_type

        if baseline_type == "value_function":
            self.value_network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            ).to(device)
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        elif baseline_type == "moving_average":
            self.baseline_value = 0.0
            self.baseline_decay = 0.95

        self.episode_states = []

        self.variance_history = []
        self.baseline_values = []

    def select_action(self, state):
        """Select action and store state for baseline computation"""
        action, log_prob = self.policy_network.get_action_and_log_prob(state)
        self.episode_log_probs.append(log_prob)
        self.episode_states.append(state)
        return action

    def calculate_baselines(self, states):
        """Calculate baselines based on chosen method"""
        if self.baseline_type == "moving_average":
            return [self.baseline_value] * len(states)

        elif self.baseline_type == "value_function":
            states_tensor = torch.FloatTensor(states).to(device)
            with torch.no_grad():
                baselines = self.value_network(states_tensor).squeeze().cpu().numpy()
            return baselines if isinstance(baselines, np.ndarray) else [baselines]

        else:  # no baseline
            return [0.0] * len(states)

    def update_baseline(self, returns):
        """Update baseline based on chosen method"""
        if self.baseline_type == "moving_average":
            episode_return = sum(self.episode_rewards)
            self.baseline_value = (
                self.baseline_decay * self.baseline_value
                + (1 - self.baseline_decay) * episode_return
            )

        elif self.baseline_type == "value_function":
            states_tensor = torch.FloatTensor(self.episode_states).to(device)
            returns_tensor = torch.FloatTensor(returns).to(device)

            predicted_values = self.value_network(states_tensor).squeeze()
            value_loss = F.mse_loss(predicted_values, returns_tensor)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def update_policy(self):
        """Update policy with baseline variance reduction"""
        if len(self.episode_log_probs) == 0:
            return

        returns = self.calculate_returns()
        returns_np = returns.cpu().numpy()

        baselines = self.calculate_baselines(self.episode_states)

        advantages = returns_np - np.array(baselines)

        self.variance_history.append(np.var(advantages))
        self.baseline_values.append(np.mean(baselines))

        policy_loss = []
        for log_prob, advantage in zip(self.episode_log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()

        total_norm = 0
        for param in self.policy_network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.gradient_norms.append(total_norm)

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_baseline(returns_np)

        self.policy_losses.append(policy_loss.item())

        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_states = []


class VarianceAnalyzer:
    """Analyze variance reduction techniques"""

    def compare_baseline_methods(self, env_name="CartPole-v1", num_episodes=300):
        """Compare different baseline methods"""

        print("=" * 70)
        print("Variance Reduction Techniques Comparison")
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        methods = {
            "No Baseline": REINFORCEAgent(state_dim, action_dim, lr=1e-3),
            "Moving Average": BaselineREINFORCEAgent(
                state_dim, action_dim, lr=1e-3, baseline_type="moving_average"
            ),
            "Value Function": BaselineREINFORCEAgent(
                state_dim, action_dim, lr=1e-3, baseline_type="value_function"
            ),
        }

        results = {}

        for method_name, agent in methods.items():
            print(f"\nTraining {method_name}...")

            episode_rewards = []

            for episode in range(num_episodes):
                reward, _ = agent.train_episode(env)
                episode_rewards.append(reward)

                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(episode_rewards[-20:])
                    print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f}")

            eval_results = agent.evaluate(env, 20)

            results[method_name] = {
                "agent": agent,
                "episode_rewards": episode_rewards,
                "final_performance": np.mean(episode_rewards[-20:]),
                "eval_performance": eval_results,
            }

        env.close()

        self.visualize_variance_comparison(results)

        return results

    def visualize_variance_comparison(self, results):
        """Visualize variance reduction comparison"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        colors = ["blue", "red", "green"]

        ax = axes[0, 0]
        for i, (method, data) in enumerate(results.items()):
            rewards = data["episode_rewards"]
            smoothed = pd.Series(rewards).rolling(window=20).mean()
            ax.plot(smoothed, label=method, color=colors[i], linewidth=2)

        ax.set_title("Learning Curves Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        baseline_methods = {k: v for k, v in results.items() if "Baseline" in k}

        for i, (method, data) in enumerate(baseline_methods.items()):
            agent = data["agent"]
            if hasattr(agent, "variance_history") and agent.variance_history:
                variance = agent.variance_history
                smoothed_var = pd.Series(variance).rolling(window=10).mean()
                ax.plot(smoothed_var, label=method, color=colors[i + 1], linewidth=2)

        ax.set_title("Advantage Variance Over Time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Advantage Variance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        ax = axes[0, 2]
        method_names = list(results.keys())
        final_perfs = [data["final_performance"] for data in results.values()]
        eval_means = [
            data["eval_performance"]["mean_reward"] for data in results.values()
        ]
        eval_stds = [
            data["eval_performance"]["std_reward"] for data in results.values()
        ]

        x = np.arange(len(method_names))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, final_perfs, width, label="Training", alpha=0.7, color=colors
        )
        bars2 = ax.bar(
            x + width / 2,
            eval_means,
            width,
            yerr=eval_stds,
            label="Evaluation",
            alpha=0.7,
            color=["dark" + c for c in ["blue", "red", "green"]],
        )

        ax.set_title("Final Performance Comparison")
        ax.set_ylabel("Average Reward")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for i, (method, data) in enumerate(results.items()):
            agent = data["agent"]
            if hasattr(agent, "gradient_norms") and agent.gradient_norms:
                grad_norms = agent.gradient_norms
                if len(grad_norms) > 10:
                    smoothed_norms = pd.Series(grad_norms).rolling(window=10).mean()
                    ax.plot(smoothed_norms, label=method, color=colors[i], linewidth=2)

        ax.set_title("Gradient Norms Evolution")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Gradient L2 Norm")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        ax = axes[1, 1]
        for i, (method, data) in enumerate(results.items()):
            agent = data["agent"]
            if hasattr(agent, "baseline_values") and agent.baseline_values:
                baseline_vals = agent.baseline_values
                ax.plot(baseline_vals, label=method, color=colors[i], linewidth=2)

        ax.set_title("Baseline Values Evolution")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Baseline Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]

        convergence_episodes = []
        method_names_conv = []

        for method, data in results.items():
            rewards = data["episode_rewards"]
            target_reward = np.max(rewards) * 0.8  # 80% of best performance

            for i in range(20, len(rewards)):
                if np.mean(rewards[i - 10 : i]) >= target_reward:
                    convergence_episodes.append(i)
                    break
            else:
                convergence_episodes.append(len(rewards))

            method_names_conv.append(method)

        bars = ax.bar(method_names_conv, convergence_episodes, color=colors, alpha=0.7)
        ax.set_title("Convergence Speed")
        ax.set_ylabel("Episodes to Convergence")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 50)
        print("VARIANCE REDUCTION SUMMARY")
        print("=" * 50)

        for method, data in results.items():
            agent = data["agent"]
            final_perf = data["final_performance"]
            eval_perf = data["eval_performance"]["mean_reward"]

            print(f"\n{method}:")
            print(f"  Final Training Performance: {final_perf:.2f}")
            print(
                f"  Evaluation Performance: {eval_perf:.2f} ± {data['eval_performance']['std_reward']:.2f}"
            )

            if hasattr(agent, "variance_history") and agent.variance_history:
                avg_variance = np.mean(agent.variance_history[-50:])
                print(f"  Average Advantage Variance (last 50): {avg_variance:.4f}")


class BaselineREINFORCEAnalyzer:
    """Analyzer for Baseline REINFORCE methods"""

    def __init__(self):
        self.variance_analyzer = VarianceAnalyzer()

    def train_and_analyze(self, env_name="CartPole-v1", num_episodes=300):
        """Train and analyze baseline REINFORCE agent"""

        print("=" * 70)
        print("Training Baseline REINFORCE Agent on", env_name)
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = BaselineREINFORCEAgent(
            state_dim, action_dim, lr=1e-3, gamma=0.99, baseline_type="value_function"
        )

        episode_rewards = []
        eval_episodes = []

        print("Starting training...")

        for episode in range(num_episodes):
            reward, _ = agent.train_episode(env)
            episode_rewards.append(reward)

            if (episode + 1) % 50 == 0:
                eval_results = agent.evaluate(env, num_episodes=10)
                eval_episodes.append((episode + 1, eval_results))
                print(
                    f"Episode {episode+1}: Train Reward = {reward:.1f}, "
                    f"Eval Reward = {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}"
                )

        env.close()

        final_eval = agent.evaluate(env, num_episodes=20)

        self.visualize_baseline_training(episode_rewards, eval_episodes, agent)

        print(f"\nTraining Statistics:")
        print(f"  Total Episodes: {num_episodes}")
        print(f"  Final Average Reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")
        print(f"  Best Episode Reward: {np.max(episode_rewards):.2f}")
        if hasattr(agent, "policy_losses") and agent.policy_losses:
            print(f"  Average Policy Loss: {np.mean(agent.policy_losses):.4f}")
        if hasattr(agent, "gradient_norms") and agent.gradient_norms:
            print(f"  Average Gradient Norm: {np.mean(agent.gradient_norms):.4f}")

        return agent

    def visualize_baseline_training(self, episode_rewards, eval_episodes, agent):
        """Visualize baseline training results"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        ax = axes[0, 0]
        smoothed_rewards = pd.Series(episode_rewards).rolling(window=20).mean()
        ax.plot(smoothed_rewards, color="blue", linewidth=2, label="Training")
        ax.set_title("Baseline REINFORCE Training Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.grid(True, alpha=0.3)
        ax.axhline(
            y=195, color="red", linestyle="--", alpha=0.7, label="CartPole-v1 Target"
        )
        ax.legend()

        ax = axes[0, 1]
        eval_x = [ep[0] for ep in eval_episodes]
        eval_y = [ep[1]["mean_reward"] for ep in eval_episodes]
        eval_std = [ep[1]["std_reward"] for ep in eval_episodes]

        ax.plot(
            eval_x, eval_y, "o-", color="green", linewidth=2, label="Evaluation Mean"
        )
        ax.fill_between(
            eval_x,
            np.array(eval_y) - np.array(eval_std),
            np.array(eval_y) + np.array(eval_std),
            alpha=0.3,
            color="green",
        )
        ax.set_title("Evaluation Performance")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Evaluation Reward")
        ax.grid(True, alpha=0.3)
        ax.axhline(
            y=195, color="red", linestyle="--", alpha=0.7, label="CartPole-v1 Target"
        )
        ax.legend()

        ax = axes[1, 0]
        if hasattr(agent, "variance_history") and agent.variance_history:
            variance_smoothed = (
                pd.Series(agent.variance_history).rolling(window=10).mean()
            )
            ax.plot(variance_smoothed, color="orange", linewidth=2)
            ax.set_title("Advantage Variance Over Time")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Advantage Variance")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        ax = axes[1, 1]
        if hasattr(agent, "baseline_values") and agent.baseline_values:
            ax.plot(agent.baseline_values, color="purple", linewidth=2)
            ax.set_title("Baseline Values Evolution")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Baseline Value")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        plt.savefig(
            "visualizations/baseline_reinforce_training.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Training visualizations saved to 'visualizations/baseline_reinforce_training.png'"
        )
