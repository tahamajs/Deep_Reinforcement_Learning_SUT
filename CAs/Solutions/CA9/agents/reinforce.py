import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """Simple policy network for discrete action spaces"""

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


class REINFORCEAgent:
    """REINFORCE Algorithm Implementation"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        self.policy_network = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        self.episode_log_probs = []
        self.episode_rewards = []

        self.episode_rewards_history = []
        self.policy_losses = []
        self.gradient_norms = []

    def select_action(self, state):
        """Select action based on current policy"""
        action, log_prob = self.policy_network.get_action_and_log_prob(state)
        self.episode_log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        """Store reward for current episode"""
        self.episode_rewards.append(reward)

    def calculate_returns(self):
        """Calculate discounted returns for the episode"""
        returns = []
        discounted_sum = 0

        for reward in reversed(self.episode_rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.FloatTensor(returns).to(device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        if len(self.episode_log_probs) == 0:
            return

        returns = self.calculate_returns()

        policy_loss = []
        for log_prob, G_t in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G_t)  # Negative for gradient ascent

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

        self.policy_losses.append(policy_loss.item())

        self.episode_log_probs = []
        self.episode_rewards = []

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

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }


class REINFORCEAnalyzer:
    """Analysis tools for REINFORCE algorithm"""

    def train_and_analyze(self, env_name="CartPole-v1", num_episodes=500):
        """Train REINFORCE agent and analyze performance"""

        print("=" * 70)
        print(f"Training REINFORCE Agent on {env_name}")
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = REINFORCEAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

        print("Starting training...")

        for episode in range(num_episodes):
            reward, steps = agent.train_episode(env)

            if (episode + 1) % 50 == 0:
                eval_results = agent.evaluate(env, 10)
                print(
                    f"Episode {episode+1}: "
                    f"Train Reward = {reward:.1f}, "
                    f"Eval Reward = {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}"
                )

        env.close()

        self.analyze_training_dynamics(agent, env_name)

        return agent

    def analyze_training_dynamics(self, agent, env_name):
        """Analyze training dynamics of REINFORCE"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        ax = axes[0, 0]
        rewards = agent.episode_rewards_history

        if len(rewards) > 10:
            smoothed_rewards = pd.Series(rewards).rolling(window=20).mean()
            ax.plot(rewards, alpha=0.3, color="lightblue", label="Episode Rewards")
            ax.plot(
                smoothed_rewards,
                color="blue",
                linewidth=2,
                label="Smoothed (20-episode avg)",
            )
        else:
            ax.plot(rewards, color="blue", linewidth=2, label="Episode Rewards")

        ax.set_title(f"REINFORCE Learning Curve - {env_name}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if agent.policy_losses:
            losses = agent.policy_losses
            ax.plot(losses, color="red", alpha=0.7)
            if len(losses) > 20:
                smoothed_losses = pd.Series(losses).rolling(window=20).mean()
                ax.plot(smoothed_losses, color="darkred", linewidth=2, label="Smoothed")
                ax.legend()

            ax.set_title("Policy Loss Evolution")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Policy Loss")
            ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if agent.gradient_norms:
            grad_norms = agent.gradient_norms
            ax.plot(grad_norms, color="green", alpha=0.7)
            if len(grad_norms) > 20:
                smoothed_norms = pd.Series(grad_norms).rolling(window=20).mean()
                ax.plot(
                    smoothed_norms, color="darkgreen", linewidth=2, label="Smoothed"
                )
                ax.legend()

            ax.set_title("Gradient Norms")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Gradient L2 Norm")
            ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if len(rewards) > 50:
            n_episodes = len(rewards)
            quartile_size = n_episodes // 4

            quartiles_data = []
            quartile_labels = []

            for i in range(4):
                start_idx = i * quartile_size
                end_idx = (i + 1) * quartile_size if i < 3 else n_episodes
                quartile_rewards = rewards[start_idx:end_idx]
                quartiles_data.append(quartile_rewards)
                quartile_labels.append(f"Episodes {start_idx+1}-{end_idx}")

            ax.boxplot(quartiles_data, labels=quartile_labels)
            ax.set_title("Reward Distribution Over Training")
            ax.set_ylabel("Episode Reward")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        print(f"\nTraining Statistics:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Final Average Reward (last 50): {np.mean(rewards[-50:]):.2f}")
        print(f"  Best Episode Reward: {np.max(rewards):.2f}")
        print(
            f"  Average Policy Loss: {np.mean(agent.policy_losses) if agent.policy_losses else 'N/A':.4f}"
        )
        print(
            f"  Average Gradient Norm: {np.mean(agent.gradient_norms) if agent.gradient_norms else 'N/A':.4f}"
        )
