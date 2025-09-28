# Continuous Control with Policy Gradients
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from setup import device
import gymnasium as gym


class GaussianPolicyNet(nn.Module):
    """
    Gaussian policy network for continuous action spaces
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GaussianPolicyNet, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean and log std for Gaussian policy
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        shared_out = self.shared(x)
        mean = self.mean_head(shared_out)
        log_std = self.log_std_head(shared_out)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp for stability
        return mean, log_std

    def get_distribution(self, state):
        """Get action distribution for a state"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return distributions.Normal(mean, std)

    def get_action(self, state, return_log_prob=False):
        """Sample action from policy"""
        dist = self.get_distribution(state)
        action = dist.rsample()  # Reparameterized sampling

        if return_log_prob:
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob

        return action


class ContinuousREINFORCEAgent:
    """
    REINFORCE algorithm for continuous action spaces
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = GaussianPolicyNet(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Logging
        self.episode_rewards = []
        self.policy_losses = []
        self.log_probs = []
        self.entropy_history = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.policy_net.get_action(state, return_log_prob)

        if return_log_prob:
            action, log_prob = action
            return action.squeeze(0).cpu().numpy(), log_prob.item()

        return action.squeeze(0).cpu().numpy()

    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return returns

    def update_policy(self, states, actions, returns):
        """Update policy using REINFORCE"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Get log probabilities
        dist = self.policy_net.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # REINFORCE loss
        policy_loss = -(log_probs * returns.unsqueeze(-1)).mean()

        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Logging
        self.policy_losses.append(policy_loss.item())
        self.log_probs.extend(log_probs.detach().cpu().numpy().flatten())
        entropy = dist.entropy().mean()
        self.entropy_history.append(entropy.item())

        return policy_loss.item()

    def train_episode(self, env):
        """Train on single episode"""
        state, _ = env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        episode_reward = 0

        while True:
            action, log_prob = self.select_action(state, return_log_prob=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            episode_reward += reward

            state = next_state

            if terminated or truncated:
                break

        # Compute returns and update
        returns = self.compute_returns(rewards)
        loss = self.update_policy(states, actions, returns)

        self.episode_rewards.append(episode_reward)

        return episode_reward, loss


class ContinuousActorCriticAgent:
    """
    Actor-Critic for continuous action spaces
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.95,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor (policy)
        self.actor_net = GaussianPolicyNet(state_dim, action_dim, hidden_dim).to(device)

        # Critic (value function)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)

        # Logging
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_history = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.actor_net.get_action(state, return_log_prob)

        if return_log_prob:
            action, log_prob = action
            return action.squeeze(0).cpu().numpy(), log_prob.item()

        return action.squeeze(0).cpu().numpy()

    def get_value(self, state):
        """Get state value estimate"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            value = self.critic_net(state)

        return value.item()

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = next_values[t]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * gae
            advantages.insert(0, gae)

        return advantages

    def update_actor_critic(self, states, actions, rewards, next_states, dones):
        """Update actor and critic"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get values
        values = self.critic_net(states).squeeze()
        next_values = self.critic_net(next_states).squeeze()

        # Compute TD targets and advantages
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = (td_targets - values).detach()

        # Critic loss
        critic_loss = F.mse_loss(values, td_targets)

        # Actor loss
        dist = self.actor_net.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        actor_loss = -(log_probs * advantages.unsqueeze(-1)).mean()

        # Add entropy bonus
        entropy = dist.entropy().mean()
        actor_loss -= 0.01 * entropy

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Logging
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_history.append(entropy.item())

        return actor_loss.item(), critic_loss.item()

    def train_episode(self, env):
        """Train on single episode"""
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

        # Update networks
        actor_loss, critic_loss = self.update_actor_critic(
            states, actions, rewards, next_states, dones
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, actor_loss, critic_loss


class PPOContinuousAgent:
    """
    PPO for continuous action spaces
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.95,
        clip_ratio=0.2,
        ppo_epochs=10,
        batch_size=64,
        entropy_coef=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        # Networks
        self.actor_net = GaussianPolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        # Old policy
        self.actor_net_old = GaussianPolicyNet(state_dim, action_dim, hidden_dim).to(
            device
        )
        self.actor_net_old.load_state_dict(self.actor_net.state_dict())

        self.optimizer = optim.Adam(
            [
                {"params": self.actor_net.parameters()},
                {"params": self.critic_net.parameters()},
            ],
            lr=lr,
        )

        # Logging
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.approx_kl_divs = []
        self.clip_fractions = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action = self.actor_net.get_action(state, return_log_prob)

        if return_log_prob:
            action, log_prob = action
            return action.squeeze(0).cpu().numpy(), log_prob.item()

        return action.squeeze(0).cpu().numpy()

    def get_value(self, state):
        """Get state value estimate"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            value = self.critic_net(state)

        return value.item()

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute GAE"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = next_values[t]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * gae
            advantages.insert(0, gae)

        return advantages

    def update_ppo(self, states, actions, old_log_probs, advantages, returns):
        """Update PPO"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Current policy
                dist = self.actor_net.get_distribution(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()

                # Old policy
                with torch.no_grad():
                    old_dist = self.actor_net_old.get_distribution(batch_states)
                    old_log_probs_batch = old_dist.log_prob(batch_actions).sum(
                        dim=-1, keepdim=True
                    )

                # Ratio
                ratio = torch.exp(log_probs - old_log_probs_batch)

                # PPO clipped objective
                surr1 = ratio * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * batch_advantages.unsqueeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Add entropy
                actor_loss -= self.entropy_coef * entropy

                # Value loss
                values = self.critic_net(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)

                # Total loss
                total_loss = actor_loss + critic_loss

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_net.parameters())
                    + list(self.critic_net.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

        # Update old policy
        self.actor_net_old.load_state_dict(self.actor_net.state_dict())

        # Approximate KL
        with torch.no_grad():
            approx_kl = torch.mean(old_log_probs - log_probs).item()

        # Clip fraction
        clip_fraction = torch.mean(
            (torch.abs(ratio - 1) > self.clip_ratio).float()
        ).item()

        # Logging
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.approx_kl_divs.append(approx_kl)
        self.clip_fractions.append(clip_fraction)

        return actor_loss.item(), critic_loss.item()

    def collect_trajectory(self, env, max_steps=1000):
        """Collect trajectory for PPO"""
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action, log_prob = self.select_action(state, return_log_prob=True)
            value = self.get_value(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(terminated or truncated)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        # Compute advantages and returns
        next_values = values[1:] + [0]
        advantages = self.compute_gae(rewards, values, next_values, dones)

        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)

        return (
            states,
            actions,
            rewards,
            log_probs,
            values,
            advantages,
            returns,
            episode_reward,
            len(states),
        )

    def train_episode(self, env):
        """Train PPO on episode"""
        trajectory = self.collect_trajectory(env)
        (
            states,
            actions,
            rewards,
            log_probs,
            values,
            advantages,
            returns,
            episode_reward,
            _,
        ) = trajectory

        # Update
        actor_loss, critic_loss = self.update_ppo(
            states, actions, log_probs, advantages, returns
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, actor_loss, critic_loss


def compare_continuous_control():
    """Compare continuous control algorithms"""
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agents
    reinforce_agent = ContinuousREINFORCEAgent(state_dim, action_dim, lr=1e-3)
    ac_agent = ContinuousActorCriticAgent(
        state_dim, action_dim, actor_lr=1e-3, critic_lr=1e-3
    )
    ppo_agent = PPOContinuousAgent(state_dim, action_dim, lr=3e-4)

    agents = {"REINFORCE": reinforce_agent, "Actor-Critic": ac_agent, "PPO": ppo_agent}

    results = {}

    print("=== Continuous Control Comparison ===")

    for name, agent in agents.items():
        print(f"\nTraining {name}...")

        # Training loop
        num_episodes = 200
        log_interval = 50

        for episode in range(num_episodes):
            episode_reward, *losses = agent.train_episode(env)

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(agent.episode_rewards[-log_interval:])
                print(".2f")

        results[name] = {
            "rewards": agent.episode_rewards.copy(),
            "losses": getattr(
                agent, "policy_losses", getattr(agent, "actor_losses", [])
            ),
        }

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, data in results.items():
        # Learning curves
        axes[0].plot(data["rewards"], label=name, alpha=0.7)
        axes[0].plot(
            pd.Series(data["rewards"]).rolling(window=20).mean(),
            linestyle="--",
            alpha=0.7,
        )

        # Losses
        axes[1].plot(data["losses"], label=name, alpha=0.7)

    axes[0].set_title("Learning Curves (Pendulum-v1)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Training Losses")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    env.close()
    return results


def test_continuous_control():
    """Test continuous control implementations"""
    return compare_continuous_control()


# Demonstration functions
def demonstrate_continuous_control():
    """Demonstrate continuous control with policy gradients"""
    print("🎛️ Continuous Control Demonstration")
    results = test_continuous_control()
    return results


# Run demonstration
if __name__ == "__main__":
    demonstrate_continuous_control()
