import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..utils.setup import device, Categorical
import gymnasium as gym


class VarianceReductionAgent:
    """
    Agent demonstrating various variance reduction techniques
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

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.variances = []
        self.baseline_values = []

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

    def get_baseline(self, state):
        """Get baseline value estimate"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            baseline = self.value_net(state)

        return baseline.item()

    def compute_returns(self, rewards):
        """Compute discounted returns (G_t values)"""
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return returns

    def compute_gae(self, rewards, values, next_values, dones, tau=0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = next_values[t]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * tau * gae
            advantages.insert(0, gae)

        return advantages

    def update_with_baseline(self, states, actions, rewards):
        """Update policy using baseline for variance reduction"""
        returns = self.compute_returns(rewards)

        states_tensor = torch.FloatTensor(states).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)

        baseline_preds = self.value_net(states_tensor).squeeze()
        value_loss = F.mse_loss(baseline_preds, returns_tensor)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        advantages = []
        for i, state in enumerate(states):
            baseline = self.get_baseline(state)
            advantage = returns[i] - baseline
            advantages.append(advantage)

        advantages_tensor = torch.FloatTensor(advantages).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        states_tensor = torch.FloatTensor(states).to(device)

        logits = self.policy_net(states_tensor)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)

        policy_loss = -(log_probs * advantages_tensor).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.baseline_values.extend([self.get_baseline(s) for s in states])

        return policy_loss.item(), value_loss.item()

    def update_with_gae(self, states, actions, rewards, next_states, dones, tau=0.95):
        """Update policy using Generalized Advantage Estimation (GAE)"""

        values = [self.get_baseline(s) for s in states]
        next_values = [self.get_baseline(s) for s in next_states] + [0]

        advantages = self.compute_gae(rewards, values, next_values[:-1], dones, tau)

        returns = [adv + val for adv, val in zip(advantages, values)]

        states_tensor = torch.FloatTensor(states).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)

        baseline_preds = self.value_net(states_tensor).squeeze()
        value_loss = F.mse_loss(baseline_preds, returns_tensor)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        advantages_tensor = torch.FloatTensor(advantages).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        states_tensor = torch.FloatTensor(states).to(device)

        logits = self.policy_net(states_tensor)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)

        policy_loss = -(log_probs * advantages_tensor).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())

        return policy_loss.item(), value_loss.item()

    def train_episode_baseline(self, env):
        """Train episode with baseline variance reduction"""
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

        policy_loss, value_loss = self.update_with_baseline(states, actions, rewards)

        self.episode_rewards.append(episode_reward)

        return episode_reward, policy_loss, value_loss

    def train_episode_gae(self, env, tau=0.95):
        """Train episode with GAE variance reduction"""
        state, _ = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_reward = 0

        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(terminated or truncated)
            episode_reward += reward

            state = next_state

            if terminated or truncated:
                break

        policy_loss, value_loss = self.update_with_gae(
            states, actions, rewards, next_states, dones, tau
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, policy_loss, value_loss


class ControlVariatesAgent:
    """
    Agent demonstrating control variates for variance reduction
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

        self.control_variates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                ).to(device)
                for _ in range(3)
            ]
        )

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.cv_optimizers = [
            optim.Adam(cv.parameters(), lr=lr) for cv in self.control_variates
        ]

        self.episode_rewards = []
        self.policy_losses = []
        self.cv_losses = []
        self.variances = []

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

    def get_control_variates(self, state):
        """Get control variate values"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            cvs = [cv(state).item() for cv in self.control_variates]

        return cvs

    def update_with_control_variates(self, states, actions, rewards):
        """Update policy using control variates for variance reduction"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        cv_losses = []
        for i, cv in enumerate(self.control_variates):
            cv_values = []
            for state in states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                cv_val = cv(state_tensor).item()
                cv_values.append(cv_val)

            cv_tensor = torch.FloatTensor(cv_values).to(device)
            returns_tensor = torch.FloatTensor(returns).to(device)

            cv_loss = F.mse_loss(cv_tensor, returns_tensor)
            cv_losses.append(cv_loss.item())

            self.cv_optimizers[i].zero_grad()
            cv_loss.backward()
            self.cv_optimizers[i].step()

        advantages = []
        for j, (state, ret) in enumerate(zip(states, returns)):
            cv_vals = self.get_control_variates(state)

            baseline = np.mean(cv_vals)
            advantage = ret - baseline
            advantages.append(advantage)

        advantages_tensor = torch.FloatTensor(advantages).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        states_tensor = torch.FloatTensor(states).to(device)

        logits = self.policy_net(states_tensor)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)

        policy_loss = -(log_probs * advantages_tensor).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.cv_losses.append(np.mean(cv_losses))

        return policy_loss.item(), np.mean(cv_losses)

    def train_episode_control_variates(self, env):
        """Train episode with control variates"""
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

        policy_loss, cv_loss = self.update_with_control_variates(
            states, actions, rewards
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, policy_loss, cv_loss


def compare_variance_reduction():
    """Compare different variance reduction techniques"""
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    baseline_agent = VarianceReductionAgent(state_dim, action_dim, lr=1e-3)
    gae_agent = VarianceReductionAgent(state_dim, action_dim, lr=1e-3)
    cv_agent = ControlVariatesAgent(state_dim, action_dim, lr=1e-3)

    agents = {
        "Baseline": baseline_agent,
        "GAE": gae_agent,
        "Control Variates": cv_agent,
    }

    methods = {
        "Baseline": "train_episode_baseline",
        "GAE": "train_episode_gae",
        "Control Variates": "train_episode_control_variates",
    }

    results = {}

    print("=== Variance Reduction Techniques Comparison ===")

    for name, agent in agents.items():
        print(f"\nTraining {name}...")

        num_episodes = 200
        log_interval = 50

        for episode in range(num_episodes):
            if name == "Control Variates":
                episode_reward, policy_loss, aux_loss = (
                    agent.train_episode_control_variates(env)
                )
            elif name == "GAE":
                episode_reward, policy_loss, aux_loss = agent.train_episode_gae(env)
            else:
                episode_reward, policy_loss, aux_loss = agent.train_episode_baseline(
                    env
                )

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(agent.episode_rewards[-log_interval:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

        results[name] = {
            "rewards": agent.episode_rewards.copy(),
            "policy_losses": agent.policy_losses.copy(),
            "aux_losses": getattr(
                agent, "value_losses", getattr(agent, "cv_losses", [])
            ),
        }

    print("\n=== Variance Analysis ===")

    variances = {}
    for name, agent in agents.items():

        variances[name] = np.var(agent.policy_losses)

    print("Gradient variance estimates:")
    for name, var in variances.items():
        print(f"{name:20} | Variance: {var:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for name, data in results.items():

        axes[0, 0].plot(data["rewards"], label=name, alpha=0.7)
        axes[0, 0].plot(
            pd.Series(data["rewards"]).rolling(window=20).mean(),
            linestyle="--",
            alpha=0.7,
        )

        axes[0, 1].plot(data["policy_losses"], label=name, alpha=0.7)

        axes[1, 0].plot(data["aux_losses"], label=name, alpha=0.7)

    names = list(variances.keys())
    vars_values = list(variances.values())
    axes[1, 1].bar(names, vars_values, alpha=0.7)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Gradient Variance Comparison")
    axes[1, 1].set_ylabel("Variance (log scale)")

    axes[0, 0].set_title("Learning Curves")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Auxiliary Loss (Value/CV)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    env.close()
    return results, variances


def test_variance_reduction():
    """Test variance reduction techniques"""
    return compare_variance_reduction()


def demonstrate_variance_reduction():
    """Demonstrate variance reduction techniques"""
    print("📊 Variance Reduction Techniques Demonstration")
    results, variances = test_variance_reduction()
    return results, variances


if __name__ == "__main__":
    demonstrate_variance_reduction()
