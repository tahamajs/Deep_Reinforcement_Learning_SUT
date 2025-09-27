import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from utils import device


class ActorNetwork(nn.Module):
    """Policy network (Actor) for discrete actions"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
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
        action_probs = self.forward(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), log_prob, entropy


class CriticNetwork(nn.Module):
    """Value network (Critic)"""

    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze()


class ActorCriticAgent:
    """Actor-Critic Algorithm Implementation"""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        entropy_coeff=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Training metrics
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.td_errors = []
        self.entropies = []
        self.value_estimates = []

    def select_action(self, state):
        """Select action using current policy"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        action, log_prob, entropy = self.actor.get_action_and_log_prob(state)
        value = self.critic(state)

        return action, log_prob, entropy, value

    def update(self, state, action, reward, next_state, done, log_prob, entropy, value):
        """Update actor and critic networks"""

        # Convert to tensors
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        if not isinstance(value, torch.Tensor):
            value = torch.FloatTensor([value]).to(device)

        # Calculate TD target and error
        with torch.no_grad():
            next_value = self.critic(next_state) if not done else 0
            td_target = reward + self.gamma * next_value
            td_error = td_target - value

        # Critic update (value function)
        value_tensor = torch.tensor(value, device=device, dtype=torch.float32)
        critic_loss = F.mse_loss(value_tensor, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update (policy)
        actor_loss = -log_prob * td_error.detach() - self.entropy_coeff * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Store metrics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.td_errors.append(td_error.item())
        self.entropies.append(entropy.item())
        self.value_estimates.append(value.item())

        return td_error.item()

    def train_episode(self, env, max_steps=1000):
        """Train for one episode"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action, log_prob, entropy, value = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update networks
            td_error = self.update(
                state, action, reward, next_state, done, log_prob, entropy, value
            )

            total_reward += reward
            steps += 1

            if done:
                break

            state = next_state

        self.episode_rewards.append(total_reward)
        return total_reward, steps

    def evaluate(self, env, num_episodes=10):
        """Evaluate current policy"""
        self.actor.eval()
        self.critic.eval()
        rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0

            for _ in range(1000):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_probs = self.actor(state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if done:
                    break

                state = next_state

            rewards.append(total_reward)

        self.actor.train()
        self.critic.train()

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }


class A2CAgent(ActorCriticAgent):
    """Advantage Actor-Critic (A2C) Implementation"""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        entropy_coeff=0.01,
        n_steps=5,
    ):
        super().__init__(
            state_dim, action_dim, lr_actor, lr_critic, gamma, entropy_coeff
        )
        self.n_steps = n_steps

        # Storage for n-step updates
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        self.dones = []

    def store_transition(self, state, action, reward, log_prob, value, entropy, done):
        """Store transition for n-step updates"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.dones.append(done)

    def compute_n_step_returns(self, next_value):
        """Compute n-step returns and advantages"""
        returns = []
        advantages = []

        # Add next_value for bootstrapping
        values = self.values + [next_value]

        # Compute n-step returns
        for i in range(len(self.rewards)):
            n_step_return = 0
            for j in range(self.n_steps):
                if i + j >= len(self.rewards):
                    break
                n_step_return += (self.gamma**j) * self.rewards[i + j]
                if self.dones[i + j]:
                    break

            # Add bootstrapped value if episode didn't end
            if i + self.n_steps < len(self.rewards) and not any(
                self.dones[i : i + self.n_steps]
            ):
                n_step_return += (self.gamma**self.n_steps) * values[i + self.n_steps]

            returns.append(n_step_return)
            advantages.append(n_step_return - values[i])

        return returns, advantages

    def update_networks(self, next_state):
        """Update networks using stored transitions"""
        if len(self.states) == 0:
            return

        # Get next state value for bootstrapping
        with torch.no_grad():
            if next_state is not None:
                next_state_tensor = (
                    torch.FloatTensor(next_state).unsqueeze(0).to(device)
                )
                next_value = self.critic(next_state_tensor).item()
            else:
                next_value = 0

        # Compute returns and advantages
        returns, advantages = self.compute_n_step_returns(next_value)

        # Convert to tensors
        states_tensor = torch.FloatTensor(self.states).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        log_probs_tensor = torch.stack(self.log_probs)
        entropies_tensor = torch.stack(self.entropies)

        # Normalize advantages
        if len(advantages) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )

        # Critic update
        values_pred = self.critic(states_tensor)
        critic_loss = F.mse_loss(values_pred, returns_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update
        actor_loss = (
            -(log_probs_tensor * advantages_tensor).mean()
            - self.entropy_coeff * entropies_tensor.mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Store metrics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.td_errors.extend(advantages)
        self.entropies.extend([e.item() for e in entropies_tensor])
        self.value_estimates.extend([v.item() for v in values_pred])

        # Clear storage
        self.clear_storage()

    def clear_storage(self):
        """Clear stored transitions"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.entropies.clear()
        self.dones.clear()

    def train_episode(self, env, max_steps=1000):
        """Train for one episode with n-step updates"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action, log_prob, entropy, value = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            self.store_transition(state, action, reward, log_prob, value, entropy, done)

            total_reward += reward
            steps += 1

            # Update networks every n_steps or at episode end
            if len(self.states) >= self.n_steps or done:
                self.update_networks(next_state if not done else None)

            if done:
                break

            state = next_state

        # Final update if there are remaining transitions
        if len(self.states) > 0:
            self.update_networks(None)

        self.episode_rewards.append(total_reward)
        return total_reward, steps


class ActorCriticAnalyzer:
    """Analyze Actor-Critic methods"""

    def compare_actor_critic_variants(self, env_name="CartPole-v1", num_episodes=300):
        """Compare different Actor-Critic variants"""

        print("=" * 70)
        print("Actor-Critic Methods Comparison")
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Different Actor-Critic variants
        agents = {
            "One-step AC": ActorCriticAgent(
                state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3
            ),
            "A2C (n=3)": A2CAgent(
                state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, n_steps=3
            ),
            "A2C (n=5)": A2CAgent(
                state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, n_steps=5
            ),
        }

        results = {}

        for name, agent in agents.items():
            print(f"\nTraining {name}...")

            for episode in range(num_episodes):
                reward, steps = agent.train_episode(env)

                if (episode + 1) % 50 == 0:
                    avg_reward = np.mean(agent.episode_rewards[-20:])
                    print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f}")

            # Evaluation
            eval_results = agent.evaluate(env, 20)

            results[name] = {
                "agent": agent,
                "final_performance": np.mean(agent.episode_rewards[-20:]),
                "eval_performance": eval_results,
            }

        env.close()

        # Visualization
        self.visualize_actor_critic_comparison(results)

        return results

    def visualize_actor_critic_comparison(self, results):
        """Visualize Actor-Critic comparison"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        colors = ["blue", "red", "green", "purple"]

        # 1. Learning curves
        ax = axes[0, 0]
        for i, (name, data) in enumerate(results.items()):
            agent = data["agent"]
            rewards = agent.episode_rewards
            smoothed = pd.Series(rewards).rolling(window=20).mean()
            ax.plot(smoothed, label=name, color=colors[i], linewidth=2)

        ax.set_title("Learning Curves")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Actor loss evolution
        ax = axes[0, 1]
        for i, (name, data) in enumerate(results.items()):
            agent = data["agent"]
            if agent.actor_losses:
                losses = agent.actor_losses
                if len(losses) > 20:
                    smoothed = pd.Series(losses).rolling(window=50).mean()
                    ax.plot(smoothed, label=name, color=colors[i], linewidth=2)

        ax.set_title("Actor Loss Evolution")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Actor Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Critic loss evolution
        ax = axes[0, 2]
        for i, (name, data) in enumerate(results.items()):
            agent = data["agent"]
            if agent.critic_losses:
                losses = agent.critic_losses
                if len(losses) > 20:
                    smoothed = pd.Series(losses).rolling(window=50).mean()
                    ax.plot(smoothed, label=name, color=colors[i], linewidth=2)

        ax.set_title("Critic Loss Evolution")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Critic Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # 4. TD errors
        ax = axes[1, 0]
        for i, (name, data) in enumerate(results.items()):
            agent = data["agent"]
            if agent.td_errors:
                td_errors = np.abs(agent.td_errors)  # Absolute TD errors
                if len(td_errors) > 50:
                    smoothed = pd.Series(td_errors).rolling(window=100).mean()
                    ax.plot(smoothed, label=name, color=colors[i], linewidth=2)

        ax.set_title("TD Error Evolution (Absolute)")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("|TD Error|")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # 5. Entropy evolution (exploration)
        ax = axes[1, 1]
        for i, (name, data) in enumerate(results.items()):
            agent = data["agent"]
            if agent.entropies:
                entropies = agent.entropies
                if len(entropies) > 50:
                    smoothed = pd.Series(entropies).rolling(window=100).mean()
                    ax.plot(smoothed, label=name, color=colors[i], linewidth=2)

        ax.set_title("Policy Entropy Evolution")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Entropy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Performance comparison
        ax = axes[1, 2]
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
            x - width / 2,
            final_perfs,
            width,
            label="Training",
            alpha=0.7,
            color=colors[: len(method_names)],
        )
        bars2 = ax.bar(
            x + width / 2,
            eval_means,
            width,
            yerr=eval_stds,
            label="Evaluation",
            alpha=0.7,
        )

        ax.set_title("Final Performance Comparison")
        ax.set_ylabel("Average Reward")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 50)
        print("ACTOR-CRITIC COMPARISON SUMMARY")
        print("=" * 50)

        for name, data in results.items():
            final_perf = data["final_performance"]
            eval_perf = data["eval_performance"]["mean_reward"]
            eval_std = data["eval_performance"]["std_reward"]

            print(f"\n{name}:")
            print(f"  Final Training Performance: {final_perf:.2f}")
            print(f"  Evaluation Performance: {eval_perf:.2f} ± {eval_std:.2f}")
