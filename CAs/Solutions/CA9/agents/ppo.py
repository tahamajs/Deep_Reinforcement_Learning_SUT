import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from .actor_critic import ActorNetwork, CriticNetwork, ActorCriticAgent, A2CAgent
from .reinforce import REINFORCEAgent
from .baseline_reinforce import BaselineREINFORCEAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOBuffer:
    """Experience buffer for PPO"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def store(self, state, action, reward, value, log_prob, done):
        """Store experience"""
        if len(self.states) >= self.capacity:
            self.clear()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae_advantages(self, last_value, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        values = self.values + [last_value]
        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]

    def get_batch(self):
        """Get all stored experiences as tensors"""
        return (
            torch.FloatTensor(self.states).to(device),
            torch.LongTensor(self.actions).to(device),
            torch.FloatTensor(self.advantages).to(device),
            torch.FloatTensor(self.returns).to(device),
            torch.FloatTensor(self.log_probs).to(device),
        )

    def clear(self):
        """Clear buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """Proximal Policy Optimization Agent"""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        buffer_size=2048,
        batch_size=64,
        entropy_coeff=0.01,
        value_coeff=0.5,
        gae_lambda=0.95,
    ):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.buffer = PPOBuffer(buffer_size)

        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.clip_fractions = []
        self.kl_divergences = []

    def select_action(self, state):
        """Select action with current policy"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, entropy = self.actor.get_action_and_log_prob(state)
            value = self.critic(state)

        return action, log_prob.item(), value.item()

    def update(self):
        """PPO update using collected experiences"""
        if len(self.buffer) < self.batch_size:
            return

        last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(device)
        with torch.no_grad():
            last_value = (
                self.critic(last_state).item() if not self.buffer.dones[-1] else 0
            )

        self.buffer.compute_gae_advantages(last_value, self.gamma, self.gae_lambda)

        states, actions, advantages, returns, old_log_probs = self.buffer.get_batch()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.k_epochs):

            batch_indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = batch_indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                action_probs = self.actor(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                values = self.critic(batch_states).squeeze()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                    * batch_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)

                entropy_loss = -entropy

                total_loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.eps_clip).float().mean()
                    self.clip_fractions.append(clip_fraction.item())

                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    self.kl_divergences.append(kl_div.item())

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropy_losses.append(entropy_loss.item())
        self.total_losses.append(total_loss.item())

        self.buffer.clear()

    def train_episode(self, env, max_steps=1000):
        """Collect experience and potentially update"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while steps < max_steps:
            action, log_prob, value = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.store(state, action, reward, value, log_prob, done)

            total_reward += reward
            steps += 1

            if len(self.buffer) >= self.buffer.capacity or done:
                self.update()
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


class AdvancedPolicyGradientAnalyzer:
    """Analyze advanced policy gradient methods"""

    def compare_all_methods(self, env_name="CartPole-v1", num_episodes=200):
        """Compare all policy gradient methods"""

        print("=" * 70)
        print("Comprehensive Policy Gradient Methods Comparison")
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        methods = {
            "REINFORCE": REINFORCEAgent(state_dim, action_dim, lr=1e-3),
            "REINFORCE + Baseline": BaselineREINFORCEAgent(
                state_dim, action_dim, lr=1e-3, baseline_type="value_function"
            ),
            "Actor-Critic": ActorCriticAgent(
                state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3
            ),
            "A2C": A2CAgent(
                state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, n_steps=5
            ),
            "PPO": PPOAgent(state_dim, action_dim, lr=3e-4, buffer_size=1024),
        }

        results = {}

        for name, agent in methods.items():
            print(f"\nTraining {name}...")

            for episode in range(num_episodes):
                reward, steps = agent.train_episode(env)

                if (episode + 1) % 40 == 0:
                    avg_reward = np.mean(agent.episode_rewards[-10:])
                    print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.1f}")

            eval_results = agent.evaluate(env, 15)

            results[name] = {
                "agent": agent,
                "final_performance": np.mean(agent.episode_rewards[-10:]),
                "eval_performance": eval_results,
                "training_stability": (
                    np.std(agent.episode_rewards[-20:])
                    if len(agent.episode_rewards) >= 20
                    else 0
                ),
            }

        env.close()

        self.visualize_comprehensive_comparison(results)

        return results

    def visualize_comprehensive_comparison(self, results):
        """Visualize comprehensive comparison"""

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        colors = ["blue", "red", "green", "orange", "purple"]

        ax = axes[0, 0]
        for i, (name, data) in enumerate(results.items()):
            agent = data["agent"]
            rewards = agent.episode_rewards
            if len(rewards) > 5:
                smoothed = pd.Series(rewards).rolling(window=10).mean()
                ax.plot(smoothed, label=name, color=colors[i], linewidth=2)

        ax.set_title("Learning Curves Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        threshold = 450  # CartPole threshold
        convergence_episodes = []
        method_names = []

        for name, data in results.items():
            agent = data["agent"]
            rewards = agent.episode_rewards

            for i in range(10, len(rewards)):
                if np.mean(rewards[i - 5 : i]) >= threshold:
                    convergence_episodes.append(i)
                    break
            else:
                convergence_episodes.append(len(rewards))

            method_names.append(name)

        bars = ax.bar(method_names, convergence_episodes, color=colors, alpha=0.7)
        ax.set_title("Sample Efficiency (Episodes to Convergence)")
        ax.set_ylabel("Episodes")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
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
        )

        ax.set_title("Final Performance Comparison")
        ax.set_ylabel("Average Reward")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        stabilities = [data["training_stability"] for data in results.values()]
        bars = ax.bar(method_names, stabilities, color=colors, alpha=0.7)
        ax.set_title("Training Stability (Lower = More Stable)")
        ax.set_ylabel("Standard Deviation of Recent Rewards")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        loss_methods = {}

        for name, data in results.items():
            agent = data["agent"]
            if hasattr(agent, "policy_losses") and agent.policy_losses:
                loss_methods[name] = agent.policy_losses
            elif hasattr(agent, "total_losses") and agent.total_losses:
                loss_methods[name] = agent.total_losses

        for i, (name, losses) in enumerate(loss_methods.items()):
            if len(losses) > 10:
                smoothed = pd.Series(losses).rolling(window=20).mean()
                ax.plot(smoothed, label=name, color=colors[i], linewidth=2)

        ax.set_title("Policy Loss Evolution")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        ax = axes[1, 2]
        ppo_agent = None
        for name, data in results.items():
            if "PPO" in name:
                ppo_agent = data["agent"]
                break

        if (
            ppo_agent
            and hasattr(ppo_agent, "clip_fractions")
            and ppo_agent.clip_fractions
        ):
            clip_fractions = ppo_agent.clip_fractions
            kl_divs = ppo_agent.kl_divergences

            ax2 = ax.twinx()

            line1 = ax.plot(
                clip_fractions, color="blue", linewidth=2, label="Clip Fraction"
            )
            line2 = ax2.plot(kl_divs, color="red", linewidth=2, label="KL Divergence")

            ax.set_xlabel("Update Step")
            ax.set_ylabel("Clip Fraction", color="blue")
            ax2.set_ylabel("KL Divergence", color="red")
            ax.set_title("PPO Training Metrics")

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper right")

        ax.grid(True, alpha=0.3)

        ax = axes[2, 0]

        characteristics = [
            "Sample Efficiency",
            "Stability",
            "Implementation Complexity",
            "Convergence Speed",
            "Final Performance",
        ]

        scores = {
            "REINFORCE": [2, 2, 5, 2, 3],
            "REINFORCE + Baseline": [3, 3, 4, 3, 3],
            "Actor-Critic": [3, 3, 3, 4, 4],
            "A2C": [4, 4, 3, 4, 4],
            "PPO": [5, 5, 2, 4, 5],
        }

        heatmap_data = np.array([scores[method] for method in method_names])
        im = ax.imshow(heatmap_data.T, cmap="RdYlGn", aspect="auto", vmin=1, vmax=5)

        ax.set_xticks(np.arange(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45)
        ax.set_yticks(np.arange(len(characteristics)))
        ax.set_yticklabels(characteristics)

        for i in range(len(method_names)):
            for j in range(len(characteristics)):
                text = ax.text(
                    i,
                    j,
                    heatmap_data[i, j],
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_title("Method Characteristics (1-5 Scale)")

        ax = axes[2, 1]
        complexities = [5, 4, 3, 3, 2]  # Lower is better
        performances = final_perfs

        scatter = ax.scatter(
            complexities, performances, c=colors[: len(method_names)], s=100, alpha=0.7
        )

        for i, name in enumerate(method_names):
            ax.annotate(
                name,
                (complexities[i], performances[i]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("Implementation Complexity (Lower = Simpler)")
        ax.set_ylabel("Final Performance")
        ax.set_title("Performance vs Implementation Complexity")
        ax.grid(True, alpha=0.3)

        ax = axes[2, 2]
        ax.axis("off")

        summary_text = (
            ".2f"
            ".2f"
            ".2f"
            ".2f"
            ".2f"
            f"""
Policy Gradient Methods Summary:

Best Sample Efficiency: {method_names[np.argmin(convergence_episodes)]}
Best Final Performance: {method_names[np.argmax(final_perfs)]}
Most Stable: {method_names[np.argmin(stabilities)]}
Simplest Implementation: REINFORCE
Most Advanced: PPO

Key Insights:
• PPO offers best overall performance
• Actor-Critic methods provide good balance
• Variance reduction significantly improves stability
• More complex methods generally perform better
"""
        )

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 60)
        print("COMPREHENSIVE POLICY GRADIENT ANALYSIS")
        print("=" * 60)

        for name, data in results.items():
            agent = data["agent"]
            final_perf = data["final_performance"]
            eval_perf = data["eval_performance"]["mean_reward"]
            stability = data["training_stability"]

            print(f"\n{name}:")
            print(
                f"  Episodes to Convergence: {convergence_episodes[list(method_names).index(name)]}"
            )
            print(f"  Final Training Performance: {final_perf:.2f}")
            print(
                f"  Evaluation Performance: {eval_perf:.2f} ± {data['eval_performance']['std_reward']:.2f}"
            )
            print(f"  Training Stability (std): {stability:.2f}")

            if hasattr(agent, "policy_losses") and agent.policy_losses:
                print(f"  Final Policy Loss: {np.mean(agent.policy_losses[-10:]):.4f}")
            if hasattr(agent, "total_losses") and agent.total_losses:
                print(f"  Final Total Loss: {np.mean(agent.total_losses[-10:]):.4f}")
