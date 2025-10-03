import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.setup import device, Categorical
import gymnasium as gym
import multiprocessing as mp
from multiprocessing import Pipe, Process
import threading
import time


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) implementation
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        tau=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

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

        self.optimizer = optim.Adam(
            [
                {"params": self.actor_net.parameters()},
                {"params": self.critic_net.parameters()},
            ],
            lr=lr,
        )

        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_history = []
        self.value_estimates = []

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

    def update_a2c(self, states, actions, rewards, next_states, dones):
        """Update A2C using collected trajectories"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        values = self.critic_net(states).squeeze()
        next_values = self.critic_net(next_states).squeeze()

        advantages = torch.FloatTensor(
            self.compute_gae(
                rewards.cpu().numpy(),
                values.detach().cpu().numpy(),
                next_values.detach().cpu().numpy(),
                dones.cpu().numpy(),
            )
        ).to(device)

        with torch.no_grad():
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        td_targets = advantages + values

        critic_loss = F.mse_loss(values, td_targets.detach())

        logits = self.actor_net(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        actor_loss = -(log_probs * advantages.detach()).mean()

        entropy = dist.entropy().mean()
        actor_loss -= self.entropy_coef * entropy

        total_loss = actor_loss + self.value_coef * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor_net.parameters()) + list(self.critic_net.parameters()),
            max_norm=0.5,
        )
        self.optimizer.step()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_history.append(entropy.item())
        self.value_estimates.extend(values.detach().cpu().numpy())

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

        actor_loss, critic_loss = self.update_a2c(
            states, actions, rewards, next_states, dones
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, actor_loss, critic_loss


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) implementation
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
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

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

        self.actor_net_old = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.actor_net_old.load_state_dict(self.actor_net.state_dict())

        self.optimizer = optim.Adam(
            [
                {"params": self.actor_net.parameters()},
                {"params": self.critic_net.parameters()},
            ],
            lr=lr,
        )

        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.approx_kl_divs = []
        self.clip_fractions = []
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

    def update_ppo(self, states, actions, old_log_probs, advantages, returns):
        """Update PPO using collected trajectories"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)

        with torch.no_grad():
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

                logits = self.actor_net(batch_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                with torch.no_grad():
                    old_logits = self.actor_net_old(batch_states)
                    old_dist = Categorical(logits=old_logits)
                    old_log_probs_batch = old_dist.log_prob(batch_actions)

                ratio = torch.exp(log_probs - old_log_probs_batch)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic_net(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)

                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_net.parameters())
                    + list(self.critic_net.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

        self.actor_net_old.load_state_dict(self.actor_net.state_dict())

        with torch.no_grad():
            approx_kl = torch.mean(batch_old_log_probs - log_probs).item()

        clip_fraction = torch.mean(
            (torch.abs(ratio - 1) > self.clip_ratio).float()
        ).item()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.approx_kl_divs.append(approx_kl)
        self.clip_fractions.append(clip_fraction)
        self.entropy_history.append(entropy.item())

        return actor_loss.item(), critic_loss.item()

    def collect_trajectory(self, env, max_steps=1000):
        """Collect single trajectory for PPO training"""
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
        """Train PPO on single episode"""
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

        actor_loss, critic_loss = self.update_ppo(
            states, actions, log_probs, advantages, returns
        )

        self.episode_rewards.append(episode_reward)

        return episode_reward, actor_loss, critic_loss


class A3CWorker(mp.Process):
    """A3C worker process"""

    def __init__(
        self,
        global_actor,
        global_critic,
        optimizer,
        env_name,
        worker_id,
        gamma=0.99,
        tau=0.95,
        t_max=20,
    ):
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.env_name = env_name
        self.gamma = gamma
        self.tau = tau
        self.t_max = t_max

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.optimizer = optimizer

        self.local_actor = nn.Sequential(
            nn.Linear(global_actor[0].in_features, global_actor[0].out_features),
            nn.ReLU(),
            nn.Linear(global_actor[2].in_features, global_actor[2].out_features),
            nn.ReLU(),
            nn.Linear(global_actor[4].in_features, global_actor[4].out_features),
        ).to(device)

        self.local_critic = nn.Sequential(
            nn.Linear(global_critic[0].in_features, global_critic[0].out_features),
            nn.ReLU(),
            nn.Linear(global_critic[2].in_features, global_critic[2].out_features),
            nn.ReLU(),
            nn.Linear(global_critic[4].in_features, global_critic[4].out_features),
        ).to(device)

        self.sync_with_global()

    def sync_with_global(self):
        """Sync local networks with global networks"""
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    def push_to_global(self):
        """Push gradients to global networks"""
        for global_param, local_param in zip(
            self.global_actor.parameters(), self.local_actor.parameters()
        ):
            if global_param.grad is None:
                global_param.grad = local_param.grad
            else:
                global_param.grad += local_param.grad

        for global_param, local_param in zip(
            self.global_critic.parameters(), self.local_critic.parameters()
        ):
            if global_param.grad is None:
                global_param.grad = local_param.grad
            else:
                global_param.grad += local_param.grad

    def select_action(self, state):
        """Select action using local policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.local_actor(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def get_value(self, state):
        """Get value using local critic"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            value = self.local_critic(state)

        return value.item()

    def run(self):
        """Main worker loop"""
        env = gym.make(self.env_name)
        episode_count = 0

        while True:
            self.sync_with_global()

            states, actions, rewards, log_probs, values = [], [], [], [], []
            state, _ = env.reset()

            done = False
            episode_reward = 0
            t = 0

            while not done and t < self.t_max:
                action, log_prob = self.select_action(state)
                value = self.get_value(state)

                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                episode_reward += reward
                state = next_state
                done = terminated or truncated
                t += 1

            if done:
                R = 0
            else:
                R = self.get_value(state)

            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            returns = torch.FloatTensor(returns).to(device)
            log_probs = torch.FloatTensor(log_probs).to(device)
            values = torch.FloatTensor(values).to(device)

            advantages = returns - values

            actor_loss = -(log_probs * advantages.detach()).mean()

            critic_loss = F.mse_loss(values, returns)

            total_loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            self.push_to_global()

            self.optimizer.step()

            episode_count += 1

            if episode_count % 10 == 0:
                print(
                    f"Worker {self.worker_id}: Episode {episode_count}, Reward: {episode_reward}"
                )

        env.close()


class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic (A3C) implementation
    """

    def __init__(
        self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99, num_workers=4
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.num_workers = num_workers

        self.global_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.global_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        self.optimizer = optim.Adam(
            [
                {"params": self.global_actor.parameters()},
                {"params": self.global_critic.parameters()},
            ],
            lr=lr,
        )

        self.workers = []
        self.episode_rewards = []

    def train(self, env_name, max_episodes=1000):
        """Train A3C with multiple workers"""

        for i in range(self.num_workers):
            worker = A3CWorker(
                self.global_actor,
                self.global_critic,
                self.optimizer,
                env_name,
                i,
                self.gamma,
            )
            worker.start()
            self.workers.append(worker)

        try:
            time.sleep(60)
        except KeyboardInterrupt:
            pass

        for worker in self.workers:
            worker.terminate()
            worker.join()

    def select_action(self, state):
        """Select action using global policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.global_actor(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()

        return action.item()


def compare_advanced_pg():
    """Compare A2C, PPO, and A3C algorithms"""
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    a2c_agent = A2CAgent(state_dim, action_dim, lr=1e-3)
    ppo_agent = PPOAgent(state_dim, action_dim, lr=3e-4)

    agents = {"A2C": a2c_agent, "PPO": ppo_agent}

    results = {}

    print("=== Advanced Policy Gradient Comparison ===")

    for name, agent in agents.items():
        print(f"\nTraining {name}...")

        num_episodes = 200
        log_interval = 50

        for episode in range(num_episodes):
            episode_reward, actor_loss, critic_loss = agent.train_episode(env)

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(agent.episode_rewards[-log_interval:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

        results[name] = {
            "rewards": agent.episode_rewards.copy(),
            "actor_losses": agent.actor_losses.copy(),
            "critic_losses": agent.critic_losses.copy(),
        }

    print("\nSkipping A3C training due to multiprocessing environment constraints...")
    print("A3C implementation is available but requires specific environment setup.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, data in results.items():

        axes[0].plot(data["rewards"], label=name, alpha=0.7)
        axes[0].plot(
            pd.Series(data["rewards"]).rolling(window=20).mean(),
            linestyle="--",
            alpha=0.7,
        )

        axes[1].plot(data["actor_losses"], label=name, alpha=0.7)

    axes[0].set_title("Learning Curves")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Actor Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    env.close()
    return results


def test_advanced_pg():
    """Test advanced policy gradient implementations"""
    return compare_advanced_pg()


def demonstrate_advanced_pg():
    """Demonstrate advanced policy gradient methods"""
    print("🚀 Advanced Policy Gradient Methods Demonstration")
    results = test_advanced_pg()
    return results


if __name__ == "__main__":
    demonstrate_advanced_pg()
