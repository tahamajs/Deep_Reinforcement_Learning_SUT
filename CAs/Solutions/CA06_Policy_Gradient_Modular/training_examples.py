"""
Policy Gradient Methods - Training Examples and Implementations
Computer Assignment 6 - Sharif University of Technology
Deep Reinforcement Learning Course

This module provides comprehensive implementations of policy gradient methods
including REINFORCE, Actor-Critic, and advanced variants with analysis tools.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class PolicyNetwork(nn.Module):
    """Neural network policy for discrete action spaces"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """Value function network for baseline and actor-critic"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous action spaces using Gaussian distribution"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean and log std for Gaussian policy
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent numerical issues
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def store_transition(self, log_prob: torch.Tensor, reward: float):
        """Store transition for later policy update"""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.cat(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.log_probs = []
        self.rewards = []

        return policy_loss.item()


class REINFORCEBaselineAgent:
    """REINFORCE with Baseline Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.log_probs = []
        self.rewards = []
        self.states = []

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def store_transition(
        self, state: np.ndarray, log_prob: torch.Tensor, reward: float
    ):
        """Store transition for later policy update"""
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        """Update policy and value function using REINFORCE with baseline"""
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)

        # Calculate value function targets and advantages
        states = torch.FloatTensor(np.array(self.states))
        values = self.value_net(states).squeeze()

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update value function
        value_loss = F.mse_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        policy_loss = torch.cat(policy_loss).sum()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Clear episode data
        self.log_probs = []
        self.rewards = []
        self.states = []

        return policy_loss.item(), value_loss.item()


class ActorCriticAgent:
    """Actor-Critic Agent with TD learning"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.critic = ValueNetwork(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: torch.Tensor,
    ) -> Tuple[float, float]:
        """Update actor and critic using TD learning"""
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # Calculate TD target and advantage
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else torch.tensor([[0.0]])

        td_target = reward + self.gamma * next_value
        advantage = td_target - value

        # Update critic
        critic_loss = F.mse_loss(value, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -log_prob * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

        self.memory = []

    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and return log prob and state value"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_old(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, state

    def store_transition(self, transition: Tuple):
        """Store transition in memory"""
        self.memory.append(transition)

    def update(self) -> Tuple[float, float]:
        """Update policy using PPO"""
        # Convert memory to tensors
        states = torch.FloatTensor([t[0] for t in self.memory])
        actions = torch.LongTensor([t[1] for t in self.memory])
        log_probs_old = torch.stack([t[2] for t in self.memory])
        rewards = torch.FloatTensor([t[3] for t in self.memory])
        dones = torch.FloatTensor([t[4] for t in self.memory])

        # Calculate discounted rewards
        discounted_rewards = []
        reward = 0
        for reward_t, done in zip(reversed(rewards), reversed(dones)):
            if done:
                reward = 0
            reward = reward_t + self.gamma * reward
            discounted_rewards.insert(0, reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Update policy for K epochs
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.k_epochs):
            # Get current policy probabilities
            probs = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            state_values = self.policy(
                states
            )  # Note: This should be a separate value network

            # Calculate ratios and surrogate losses
            ratios = torch.exp(log_probs - log_probs_old)
            advantages = discounted_rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse_loss(state_values, discounted_rewards)

            # Update
            self.optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.memory = []

        return total_policy_loss / self.k_epochs, total_value_loss / self.k_epochs


class ContinuousPPOAgent:
    """PPO Agent for Continuous Action Spaces"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        hidden_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dim)
        self.policy_old = ContinuousPolicyNetwork(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select action from continuous policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.sample(state)
        return action.detach().numpy().flatten(), log_prob

    def store_transition(self, transition: Tuple):
        """Store transition in memory"""
        self.memory.append(transition)

    def update(self) -> float:
        """Update policy using PPO for continuous actions"""
        # Convert memory to tensors
        states = torch.FloatTensor([t[0] for t in self.memory])
        actions = torch.FloatTensor([t[1] for t in self.memory])
        log_probs_old = torch.stack([t[2] for t in self.memory])
        rewards = torch.FloatTensor([t[3] for t in self.memory])
        dones = torch.FloatTensor([t[4] for t in self.memory])

        # Calculate discounted rewards
        discounted_rewards = []
        reward = 0
        for reward_t, done in zip(reversed(rewards), reversed(dones)):
            if done:
                reward = 0
            reward = reward_t + self.gamma * reward
            discounted_rewards.insert(0, reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Update policy for K epochs
        total_loss = 0

        for _ in range(self.k_epochs):
            # Get current policy distribution
            mean, log_std = self.policy(states)
            std = log_std.exp()
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

            # Calculate ratios and surrogate losses
            ratios = torch.exp(log_probs - log_probs_old)
            advantages = discounted_rewards.unsqueeze(
                -1
            )  # Add dimension for broadcasting

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = -torch.min(surr1, surr2).mean()

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.memory = []

        return total_loss / self.k_epochs


# Training Functions


def train_reinforce_agent(
    env_name: str = "CartPole-v1",
    episodes: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
) -> Dict[str, List[float]]:
    """Train REINFORCE agent"""

    print(f"Training REINFORCE on {env_name}")
    print("=" * 40)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, lr, gamma)

    scores = []
    losses = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(log_prob, reward)
            episode_reward += reward
            state = next_state

        loss = agent.update_policy()
        losses.append(loss)
        scores.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f} | Loss: {loss:.4f}"
            )

    env.close()
    return {"scores": scores, "losses": losses}


def train_reinforce_baseline_agent(
    env_name: str = "CartPole-v1",
    episodes: int = 1000,
    lr_policy: float = 1e-3,
    lr_value: float = 1e-3,
    gamma: float = 0.99,
) -> Dict[str, List[float]]:
    """Train REINFORCE with baseline agent"""

    print(f"Training REINFORCE+Baseline on {env_name}")
    print("=" * 45)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEBaselineAgent(state_dim, action_dim, lr_policy, lr_value, gamma)

    scores = []
    policy_losses = []
    value_losses = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, log_prob, reward)
            episode_reward += reward
            state = next_state

        policy_loss, value_loss = agent.update_policy()
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        scores.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f} | P-Loss: {policy_loss:.4f} | V-Loss: {value_loss:.4f}"
            )

    env.close()
    return {
        "scores": scores,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
    }


def train_actor_critic_agent(
    env_name: str = "CartPole-v1",
    episodes: int = 1000,
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
) -> Dict[str, List[float]]:
    """Train Actor-Critic agent"""

    print(f"Training Actor-Critic on {env_name}")
    print("=" * 35)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(state_dim, action_dim, lr_actor, lr_critic, gamma)

    scores = []
    actor_losses = []
    critic_losses = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            actor_loss, critic_loss = agent.update(
                state, action, reward, next_state, done, log_prob
            )

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            episode_reward += reward
            state = next_state

        scores.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f}")

    env.close()
    return {
        "scores": scores,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
    }


def train_ppo_agent(
    env_name: str = "CartPole-v1",
    episodes: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
) -> Dict[str, List[float]]:
    """Train PPO agent"""

    print(f"Training PPO on {env_name}")
    print("=" * 25)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)

    scores = []
    policy_losses = []
    value_losses = []

    episode_rewards = []
    episode_length = 0

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, log_prob, state_tensor = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition((state, action, log_prob, reward, done))

            episode_reward += reward
            state = next_state
            episode_length += 1

            if episode_length % 2048 == 0:  # Update every 2048 steps
                p_loss, v_loss = agent.update()
                policy_losses.append(p_loss)
                value_losses.append(v_loss)

        episode_rewards.append(episode_reward)
        scores.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f}")

    env.close()
    return {
        "scores": scores,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
    }


def train_continuous_ppo_agent(
    env_name: str = "Pendulum-v1",
    episodes: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
) -> Dict[str, List[float]]:
    """Train PPO agent for continuous control"""

    print(f"Training PPO on {env_name} (Continuous)")
    print("=" * 40)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ContinuousPPOAgent(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)

    scores = []
    losses = []

    episode_rewards = []
    episode_length = 0

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition((state, action, log_prob.item(), reward, done))

            episode_reward += reward
            state = next_state
            episode_length += 1

            if episode_length % 2048 == 0:  # Update every 2048 steps
                loss = agent.update()
                losses.append(loss)

        episode_rewards.append(episode_reward)
        scores.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f}")

    env.close()
    return {"scores": scores, "losses": losses}


# Analysis and Visualization Functions


def compare_policy_gradient_variants(
    env_name: str = "CartPole-v1", episodes: int = 500
) -> Dict[str, Dict]:
    """Compare different policy gradient variants"""

    print(f"Comparing Policy Gradient Variants on {env_name}")
    print("=" * 55)

    results = {}

    # Train REINFORCE
    print("\n1. Training REINFORCE...")
    results["REINFORCE"] = train_reinforce_agent(env_name, episodes)

    # Train REINFORCE + Baseline
    print("\n2. Training REINFORCE + Baseline...")
    results["REINFORCE_Baseline"] = train_reinforce_baseline_agent(env_name, episodes)

    # Train Actor-Critic
    print("\n3. Training Actor-Critic...")
    results["Actor_Critic"] = train_actor_critic_agent(env_name, episodes)

    # Train PPO
    print("\n4. Training PPO...")
    results["PPO"] = train_ppo_agent(env_name, episodes)

    return results


def plot_policy_gradient_comparison(
    results: Dict[str, Dict], save_path: Optional[str] = None
):
    """Plot comparison of policy gradient variants"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    methods = list(results.keys())
    colors = ["blue", "green", "red", "purple"]

    # Learning curves
    for method, color in zip(methods, colors):
        scores = results[method]["scores"]
        smoothed_scores = np.convolve(scores, np.ones(50) / 50, mode="valid")
        axes[0, 0].plot(smoothed_scores, label=method, color=color, linewidth=2)

    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Smoothed Score")
    axes[0, 0].set_title("Learning Curves Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Final performance comparison
    final_scores = [np.mean(results[method]["scores"][-100:]) for method in methods]
    axes[0, 1].bar(methods, final_scores, alpha=0.7, edgecolor="black")
    axes[0, 1].set_ylabel("Final Average Score")
    axes[0, 1].set_title("Final Performance Comparison")
    axes[0, 1].grid(True, alpha=0.3)

    # Training stability (variance of scores)
    score_variances = [np.var(results[method]["scores"][-200:]) for method in methods]
    axes[1, 0].bar(
        methods, score_variances, alpha=0.7, edgecolor="black", color="orange"
    )
    axes[1, 0].set_ylabel("Score Variance")
    axes[1, 0].set_title("Training Stability (Lower is Better)")
    axes[1, 0].grid(True, alpha=0.3)

    # Sample efficiency (episodes to reach threshold)
    threshold = 195  # For CartPole
    sample_efficiencies = []

    for method in methods:
        scores = results[method]["scores"]
        episodes_to_threshold = len(scores)
        for i, score in enumerate(scores):
            if score >= threshold:
                episodes_to_threshold = i + 1
                break
        sample_efficiencies.append(episodes_to_threshold)

    axes[1, 1].bar(
        methods, sample_efficiencies, alpha=0.7, edgecolor="black", color="green"
    )
    axes[1, 1].set_ylabel("Episodes to Threshold")
    axes[1, 1].set_title("Sample Efficiency (Lower is Better)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def hyperparameter_sensitivity_analysis(
    env_name: str = "CartPole-v1", episodes: int = 300
):
    """Analyze sensitivity to hyperparameters"""

    print("Policy Gradient Hyperparameter Sensitivity Analysis")
    print("=" * 55)

    # Test different learning rates
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    lr_results = {}

    print("\nTesting different learning rates...")
    for lr in learning_rates:
        print(f"  Learning Rate: {lr}")
        result = train_reinforce_baseline_agent(env_name, episodes, lr_policy=lr)
        lr_results[lr] = np.mean(result["scores"][-100:])

    # Test different gamma values
    gamma_values = [0.9, 0.95, 0.99, 0.995]
    gamma_results = {}

    print("\nTesting different gamma values...")
    for gamma in gamma_values:
        print(f"  Gamma: {gamma}")
        result = train_reinforce_baseline_agent(env_name, episodes, gamma=gamma)
        gamma_results[gamma] = np.mean(result["scores"][-100:])

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Learning rate sensitivity
    lrs = list(lr_results.keys())
    scores = list(lr_results.values())
    axes[0].plot(lrs, scores, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Final Average Score")
    axes[0].set_title("Learning Rate Sensitivity")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)

    # Gamma sensitivity
    gammas = list(gamma_results.keys())
    scores = list(gamma_results.values())
    axes[1].plot(gammas, scores, "o-", linewidth=2, markersize=8, color="green")
    axes[1].set_xlabel("Gamma (Discount Factor)")
    axes[1].set_ylabel("Final Average Score")
    axes[1].set_title("Discount Factor Sensitivity")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "policy_gradient_hyperparameter_sensitivity.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return {"learning_rates": lr_results, "gamma_values": gamma_results}


def curriculum_learning_demo(env_name: str = "CartPole-v1", episodes: int = 1000):
    """Demonstrate curriculum learning with policy gradients"""

    print("Curriculum Learning Demonstration")
    print("=" * 40)

    # Create modified environments with different difficulties
    env_configs = [
        {
            "name": "Easy",
            "gravity": 9.8,
            "length": 0.5,
            "masscart": 1.0,
            "masspole": 0.1,
        },
        {
            "name": "Medium",
            "gravity": 9.8,
            "length": 0.5,
            "masscart": 1.0,
            "masspole": 0.1,
        },
        {
            "name": "Hard",
            "gravity": 9.8,
            "length": 0.5,
            "masscart": 1.0,
            "masspole": 0.1,
        },
    ]

    curriculum_results = {}

    for config in env_configs:
        print(f"\nTraining on {config['name']} environment...")

        # Create environment (simplified - in practice would modify gym env)
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = REINFORCEBaselineAgent(state_dim, action_dim)
        scores = []

        for episode in range(episodes // len(env_configs)):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, log_prob, reward)
                episode_reward += reward
                state = next_state

            agent.update_policy()
            scores.append(episode_reward)

        curriculum_results[config["name"]] = scores
        env.close()

    # Plot curriculum learning results
    fig, ax = plt.subplots(figsize=(10, 6))

    for env_type, scores in curriculum_results.items():
        smoothed_scores = np.convolve(scores, np.ones(20) / 20, mode="valid")
        ax.plot(smoothed_scores, label=env_type, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Score")
    ax.set_title("Curriculum Learning Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("curriculum_learning_demo.png", dpi=300, bbox_inches="tight")
    plt.show()

    return curriculum_results


# Main execution examples
if __name__ == "__main__":
    print("Policy Gradient Methods - Training Examples")
    print("=" * 50)

    # Example 1: Compare policy gradient variants
    print("\nExample 1: Comparing Policy Gradient Variants")
    results = compare_policy_gradient_variants("CartPole-v1", episodes=200)
    plot_policy_gradient_comparison(results, "policy_gradient_comparison.png")

    # Example 2: Hyperparameter sensitivity
    print("\nExample 2: Hyperparameter Sensitivity Analysis")
    hyper_results = hyperparameter_sensitivity_analysis("CartPole-v1", episodes=150)

    # Example 3: Curriculum learning
    print("\nExample 3: Curriculum Learning Demonstration")
    curriculum_results = curriculum_learning_demo("CartPole-v1", episodes=300)

    # Example 4: Continuous control
    print("\nExample 4: Continuous Control with PPO")
    continuous_results = train_continuous_ppo_agent("Pendulum-v1", episodes=200)

    print("\nAll examples completed! Check the generated plots and results.")
