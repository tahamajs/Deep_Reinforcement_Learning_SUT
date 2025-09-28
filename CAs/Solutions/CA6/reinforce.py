import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from setup import device, Categorical
import gymnasium as gym


class REINFORCEAgent:
    """
    Complete REINFORCE (Monte Carlo Policy Gradient) implementation
    with detailed logging and analysis capabilities
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.episode_rewards = []
        self.policy_losses = []
        self.gradient_norms = []
        self.entropy_history = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.policy_net(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()

        if return_log_prob:
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

        return action.item()

    def get_policy_distribution(self, state):
        """Get full policy distribution for analysis"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.policy_net(state)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy().flatten()

    def compute_returns(self, rewards):
        """Compute discounted returns (G_t values)"""
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return returns

    def update_policy(self, states, actions, returns):
        """Update policy using REINFORCE algorithm"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits = self.policy_net(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        policy_loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()

        total_norm = 0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        self.gradient_norms.append(total_norm)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        entropy = dist.entropy().mean()
        self.entropy_history.append(entropy.item())
        self.policy_losses.append(policy_loss.item())

        return policy_loss.item()

    def train_episode(self, env):
        """Train on single episode"""
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        episode_reward = 0

        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward

            state = next_state

            if terminated or truncated:
                break

        returns = self.compute_returns(rewards)

        loss = self.update_policy(states, actions, returns)

        self.episode_rewards.append(episode_reward)

        return episode_reward, loss

    def analyze_variance(self, env, num_episodes=100):
        """Analyze gradient variance in REINFORCE"""
        gradient_estimates = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            states, actions, rewards = [], [], []

            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

                if terminated or truncated:
                    break

            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.LongTensor(actions).to(device)
            returns = torch.FloatTensor(self.compute_returns(rewards)).to(device)

            logits = self.policy_net(states_tensor)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_tensor)

            grad_contributions = (log_probs * returns).detach().cpu().numpy()
            gradient_estimates.extend(grad_contributions)

        return np.array(gradient_estimates)


def test_reinforce():
    """Test REINFORCE implementation"""
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

    print("=== REINFORCE Training ===")

    num_episodes = 300
    log_interval = 50

    for episode in range(num_episodes):
        episode_reward, loss = agent.train_episode(env)

        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(agent.episode_rewards[-log_interval:])
            avg_loss = np.mean(agent.policy_losses[-log_interval:])
            avg_entropy = np.mean(agent.entropy_history[-log_interval:])

            print(".2f" ".4f" ".4f")

    print("\n=== Variance Analysis ===")
    gradient_estimates = agent.analyze_variance(env, num_episodes=50)

    print("Gradient estimate statistics:")
    print(".4f")
    print(".4f")
    print(".4f")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(agent.episode_rewards)
    axes[0, 0].plot(
        pd.Series(agent.episode_rewards).rolling(window=20).mean(),
        color="red",
        label="Moving Average",
    )
    axes[0, 0].set_title("REINFORCE Learning Curve")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(agent.policy_losses)
    axes[0, 1].plot(
        pd.Series(agent.policy_losses).rolling(window=20).mean(),
        color="red",
        label="Moving Average",
    )
    axes[0, 1].set_title("Policy Loss Over Time")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Policy Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(agent.gradient_norms)
    axes[1, 0].set_title("Gradient Norms")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Gradient L2 Norm")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(gradient_estimates, bins=30, alpha=0.7, density=True)
    axes[1, 1].axvline(
        np.mean(gradient_estimates), color="red", linestyle="--", label=".3f"
    )
    axes[1, 1].set_title("Distribution of Gradient Estimates")
    axes[1, 1].set_xlabel("Gradient Value")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    env.close()
    return agent


def demonstrate_reinforce():
    """Demonstrate REINFORCE algorithm"""
    print("🧠 REINFORCE Algorithm Demonstration")
    agent = test_reinforce()
    return agent


if __name__ == "__main__":
    demonstrate_reinforce()
