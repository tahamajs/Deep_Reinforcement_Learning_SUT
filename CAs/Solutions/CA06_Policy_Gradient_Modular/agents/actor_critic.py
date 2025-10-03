import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.setup import device, Categorical
import gymnasium as gym


class ActorCriticNet(nn.Module):
    """
    Shared network for Actor-Critic architecture
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNet, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        actor_logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return actor_logits, value


class ActorCriticAgent:
    """
    Actor-Critic agent with separate actor and critic networks
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

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        self.critic_target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        self.critic_target.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)

        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.td_errors = []
        self.entropy_history = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.actor_net(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()

        if return_log_prob:
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

        return action.item()

    def get_value(self, state):
        """Get state value estimate"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            value = self.critic_net(state)

        return value.item()

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)"""
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
        """Update actor and critic using collected experience"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        values = self.critic_net(states).squeeze()
        next_values = self.critic_net(next_states).squeeze()

        td_targets = rewards + self.gamma * next_values * (1 - dones)

        critic_loss = F.mse_loss(values, td_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        advantages = (td_targets - values).detach()

        logits = self.actor_net(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        actor_loss = -(log_probs * advantages).mean()

        entropy = dist.entropy().mean()
        actor_loss -= 0.01 * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic_net.parameters()
        ):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_history.append(entropy.item())
        self.td_errors.extend((td_targets - values).detach().cpu().numpy())

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

        actor_loss, critic_loss = self.update_actor_critic(
            states, actions, rewards, next_states, dones
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, actor_loss, critic_loss


class SharedActorCriticAgent:
    """
    Actor-Critic agent with shared network architecture
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.shared_net = ActorCriticNet(state_dim, action_dim, hidden_dim).to(device)

        self.optimizer = optim.Adam(self.shared_net.parameters(), lr=lr)

        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_history = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = self.shared_net(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()

        if return_log_prob:
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

        return action.item()

    def get_value(self, state):
        """Get state value estimate"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            _, value = self.shared_net(state)

        return value.item()

    def update_shared(self, states, actions, rewards, next_states, dones):
        """Update shared network using collected experience"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        logits, values = self.shared_net(states)
        _, next_values = self.shared_net(next_states)

        values = values.squeeze()
        next_values = next_values.squeeze()

        td_targets = rewards + self.gamma * next_values * (1 - dones)

        critic_loss = F.mse_loss(values, td_targets.detach())

        advantages = (td_targets - values).detach()

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        actor_loss = -(log_probs * advantages).mean()

        entropy = dist.entropy().mean()
        actor_loss -= 0.01 * entropy

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_net.parameters(), max_norm=1.0)
        self.optimizer.step()

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

        actor_loss, critic_loss = self.update_shared(
            states, actions, rewards, next_states, dones
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, actor_loss, critic_loss


def compare_actor_critic_agents():
    """Compare separate vs shared Actor-Critic architectures"""
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    separate_agent = ActorCriticAgent(
        state_dim, action_dim, actor_lr=1e-3, critic_lr=1e-3
    )
    shared_agent = SharedActorCriticAgent(state_dim, action_dim, lr=1e-3)

    agents = {"Separate Networks": separate_agent, "Shared Network": shared_agent}

    results = {}

    print("=== Actor-Critic Architecture Comparison ===")

    for name, agent in agents.items():
        print(f"\nTraining {name}...")

        num_episodes = 300
        log_interval = 50

        for episode in range(num_episodes):
            episode_reward, actor_loss, critic_loss = agent.train_episode(env)

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(agent.episode_rewards[-log_interval:])
                avg_actor_loss = np.mean(agent.actor_losses[-log_interval:])
                avg_critic_loss = np.mean(agent.critic_losses[-log_interval:])

                print(
                    f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}"
                )

        results[name] = {
            "rewards": agent.episode_rewards.copy(),
            "actor_losses": agent.actor_losses.copy(),
            "critic_losses": agent.critic_losses.copy(),
            "entropy": agent.entropy_history.copy(),
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for name, data in results.items():

        axes[0, 0].plot(data["rewards"], label=name, alpha=0.7)
        axes[0, 0].plot(
            pd.Series(data["rewards"]).rolling(window=20).mean(),
            linestyle="--",
            alpha=0.7,
        )

        axes[0, 1].plot(data["actor_losses"], label=name, alpha=0.7)

        axes[1, 0].plot(data["critic_losses"], label=name, alpha=0.7)

        axes[1, 1].plot(data["entropy"], label=name, alpha=0.7)

    axes[0, 0].set_title("Learning Curves")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Actor Loss")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Critic Loss")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    env.close()
    return results


def test_actor_critic():
    """Test Actor-Critic implementation"""
    return compare_actor_critic_agents()


def demonstrate_actor_critic():
    """Demonstrate Actor-Critic methods"""
    print("🎭 Actor-Critic Methods Demonstration")
    results = test_actor_critic()
    return results


if __name__ == "__main__":
    demonstrate_actor_critic()
