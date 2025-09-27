import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Normal
from utils import device


class ContinuousActorNetwork(nn.Module):
    """Policy network for continuous action spaces using Gaussian distribution"""

    def __init__(self, state_dim, action_dim, action_bound=1.0, hidden_dim=128):
        super(ContinuousActorNetwork, self).__init__()
        self.action_bound = action_bound

        # Shared network
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bound mean between -1 and 1
        )

        # Standard deviation head (log std for numerical stability)
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )

        # Initialize log_std to reasonable values
        nn.init.constant_(self.log_std_head[0].bias, -0.5)  # exp(-0.5) ≈ 0.6

    def forward(self, state):
        """Forward pass returning mean and std"""
        shared_features = self.shared_network(state)

        mean = self.mean_head(shared_features) * self.action_bound
        log_std = self.log_std_head(shared_features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        return mean, std

    def get_action_and_log_prob(self, state):
        """Sample action and compute log probability"""
        mean, std = self.forward(state)

        # Create normal distribution
        normal_dist = Normal(mean, std)

        # Sample action
        action = normal_dist.rsample()  # Use rsample for reparameterization

        # Compute log probability
        log_prob = normal_dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Apply tanh squashing for bounded actions
        action_tanh = torch.tanh(action)

        # Correct log probability for tanh squashing
        # log_prob = log_prob - torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # Scale action to environment bounds
        scaled_action = action_tanh * self.action_bound

        return scaled_action, log_prob

    def get_deterministic_action(self, state):
        """Get deterministic action (mean) for evaluation"""
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_bound


class ContinuousREINFORCEAgent:
    """REINFORCE Algorithm for Continuous Action Spaces"""

    def __init__(self, state_dim, action_dim, action_bound=1.0, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.lr = lr
        self.gamma = gamma

        # Policy network
        self.policy_network = ContinuousActorNetwork(
            state_dim, action_dim, action_bound
        ).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Storage for episode
        self.episode_log_probs = []
        self.episode_rewards = []

        # Training metrics
        self.episode_rewards_history = []
        self.policy_losses = []
        self.gradient_norms = []

    def select_action(self, state):
        """Select action based on current policy"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        action, log_prob = self.policy_network.get_action_and_log_prob(state)
        self.episode_log_probs.append(log_prob)

        return action.detach().cpu().numpy().flatten()

    def store_reward(self, reward):
        """Store reward for current episode"""
        self.episode_rewards.append(reward)

    def calculate_returns(self):
        """Calculate discounted returns for the episode"""
        returns = []
        discounted_sum = 0

        # Calculate returns in reverse order
        for reward in reversed(self.episode_rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Convert to tensor and normalize
        returns = torch.FloatTensor(returns).to(device)

        # Optional: normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        if len(self.episode_log_probs) == 0:
            return

        # Calculate returns
        returns = self.calculate_returns()

        # Calculate policy loss
        policy_loss = []
        for log_prob, G_t in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G_t)  # Negative for gradient ascent

        policy_loss = torch.stack(policy_loss).sum()

        # Perform optimization step
        self.optimizer.zero_grad()
        policy_loss.backward()

        # Calculate and store gradient norm
        total_norm = 0
        for param in self.policy_network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.gradient_norms.append(total_norm)

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Store metrics
        self.policy_losses.append(policy_loss.item())

        # Clear episode data
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

        # Update policy at end of episode
        self.update_policy()

        # Store episode reward
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
                    action = self.policy_network.get_deterministic_action(state_tensor)

                action_np = action.detach().cpu().numpy().flatten()
                next_state, reward, terminated, truncated, _ = env.step(action_np)
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


class ContinuousControlAnalyzer:
    """Analysis and demonstration tools for continuous control with policy gradients"""

    def __init__(self):
        self.continuous_agents = {}

    def demonstrate_continuous_policy_network(self):
        """Demonstrate continuous policy network architecture and sampling"""

        print("=" * 70)
        print("Continuous Policy Network Demonstration")
        print("=" * 70)

        # Create network for Pendulum environment
        state_dim = 3  # Pendulum state: [cos(theta), sin(theta), theta_dot]
        action_dim = 1  # Pendulum action: torque
        action_bound = 2.0  # Pendulum action bound

        network = ContinuousActorNetwork(state_dim, action_dim, action_bound)

        print(f"Network Architecture:")
        print(f"  Input dimension: {state_dim}")
        print(f"  Output dimension: {action_dim} (mean) + {action_dim} (std)")
        print(f"  Action bound: ±{action_bound}")
        print(f"  Total parameters: {sum(p.numel() for p in network.parameters())}")

        # Demonstrate sampling
        print(f"\nSampling Demonstration:")
        network.eval()

        with torch.no_grad():
            # Sample different states
            sample_states = [
                torch.randn(1, state_dim),  # Random state
                torch.FloatTensor([[1.0, 0.0, 0.0]]),  # Upright pendulum
                torch.FloatTensor([[0.0, 1.0, 1.0]]),  # Swinging pendulum
            ]

            for i, state in enumerate(sample_states):
                mean, std = network(state)
                action, log_prob = network.get_action_and_log_prob(state)

                print(f"\nState {i+1}: {state.numpy().flatten()}")
                print(f"  Mean: {mean.numpy().flatten()}")
                print(f"  Std: {std.numpy().flatten()}")
                print(f"  Sampled action: {action.numpy().flatten()}")
                print(f"  Log probability: {log_prob.item():.4f}")

        return network

    def train_continuous_reinforce(self, env_name="Pendulum-v1", num_episodes=100):
        """Train REINFORCE on a continuous control task"""

        print("=" * 70)
        print(f"Training Continuous REINFORCE on {env_name}")
        print("=" * 70)

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = float(env.action_space.high[0])

        agent = ContinuousREINFORCEAgent(
            state_dim, action_dim, action_bound, lr=3e-4, gamma=0.99
        )

        print(f"Environment: {env_name}")
        print(
            f"State dim: {state_dim}, Action dim: {action_dim}, Action bound: ±{action_bound}"
        )
        print("Training...")

        for episode in range(num_episodes):
            reward, steps = agent.train_episode(env)

            if (episode + 1) % 10 == 0:
                eval_results = agent.evaluate(env, 5)
                print(
                    f"Episode {episode+1}: "
                    f"Train Reward = {reward:.1f}, "
                    f"Eval Reward = {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}"
                )

        env.close()

        # Store trained agent
        self.continuous_agents["reinforce"] = agent

        # Analysis
        self.analyze_continuous_training(agent, env_name)

        return agent

    def analyze_continuous_training(self, agent, env_name):
        """Analyze continuous control training"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Learning curve
        ax = axes[0, 0]
        rewards = agent.episode_rewards_history

        if len(rewards) > 5:
            smoothed_rewards = pd.Series(rewards).rolling(window=10).mean()
            ax.plot(rewards, alpha=0.3, color="lightblue", label="Episode Rewards")
            ax.plot(
                smoothed_rewards,
                color="blue",
                linewidth=2,
                label="Smoothed (10-episode avg)",
            )
        else:
            ax.plot(rewards, color="blue", linewidth=2, label="Episode Rewards")

        ax.set_title(f"Continuous REINFORCE Learning Curve - {env_name}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Policy loss
        ax = axes[0, 1]
        if agent.policy_losses:
            losses = agent.policy_losses
            ax.plot(losses, color="red", alpha=0.7)
            if len(losses) > 10:
                smoothed_losses = pd.Series(losses).rolling(window=10).mean()
                ax.plot(smoothed_losses, color="darkred", linewidth=2, label="Smoothed")
                ax.legend()

            ax.set_title("Policy Loss Evolution")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Policy Loss")
            ax.grid(True, alpha=0.3)

        # 3. Gradient norms
        ax = axes[1, 0]
        if agent.gradient_norms:
            grad_norms = agent.gradient_norms
            ax.plot(grad_norms, color="green", alpha=0.7)
            if len(grad_norms) > 10:
                smoothed_norms = pd.Series(grad_norms).rolling(window=10).mean()
                ax.plot(
                    smoothed_norms, color="darkgreen", linewidth=2, label="Smoothed"
                )
                ax.legend()

            ax.set_title("Gradient Norms")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Gradient L2 Norm")
            ax.grid(True, alpha=0.3)

        # 4. Action distribution evolution (if available)
        ax = axes[1, 1]
        # This would require storing action samples during training
        ax.text(
            0.5,
            0.5,
            "Action Distribution\nEvolution\n(Not implemented)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.7,
        )
        ax.set_title("Action Distribution Evolution")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\nContinuous Control Training Statistics:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Final Average Reward (last 20): {np.mean(rewards[-20:]):.2f}")
        print(f"  Best Episode Reward: {np.max(rewards):.2f}")
        print(
            f"  Average Policy Loss: {np.mean(agent.policy_losses) if agent.policy_losses else 'N/A':.4f}"
        )
        print(
            f"  Average Gradient Norm: {np.mean(agent.gradient_norms) if agent.gradient_norms else 'N/A':.4f}"
        )

    def compare_discrete_vs_continuous(self):
        """Compare discrete and continuous control performance"""

        print("=" * 70)
        print("Discrete vs Continuous Control Comparison")
        print("=" * 70)

        # This would compare trained agents on their respective environments
        # For now, just show conceptual comparison

        comparison_data = {
            "Discrete Control (CartPole)": {
                "Environment": "CartPole-v1",
                "Action Space": "Discrete (2 actions)",
                "Policy": "Categorical",
                "Challenges": "Balance binary actions",
                "Typical Reward": "~200-500",
            },
            "Continuous Control (Pendulum)": {
                "Environment": "Pendulum-v1",
                "Action Space": "Continuous (torque)",
                "Policy": "Gaussian",
                "Challenges": "Action bounds, exploration",
                "Typical Reward": "-200 to 0",
            },
        }

        for env_type, data in comparison_data.items():
            print(f"\n{env_type}:")
            for key, value in data.items():
                print(f"  {key}: {value}")

        print(f"\nKey Differences:")
        print(f"• Policy parameterization: Categorical vs Gaussian")
        print(f"• Action selection: argmax vs sampling")
        print(f"• Log probability computation: Different for each distribution")
        print(f"• Exploration: ε-greedy vs stochastic policy")
        print(f"• Gradient flow: Through softmax vs through distribution parameters")

    def demonstrate_action_scaling(self):
        """Demonstrate action scaling and bounds handling"""

        print("=" * 70)
        print("Action Scaling and Bounds Handling")
        print("=" * 70)

        # Create a simple continuous actor
        state_dim, action_dim, action_bound = 2, 1, 2.0
        actor = ContinuousActorNetwork(state_dim, action_dim, action_bound)

        print(f"Action bound: ±{action_bound}")
        print(f"Network outputs mean in range [-{action_bound}, {action_bound}]")

        # Demonstrate scaling
        actor.eval()
        with torch.no_grad():
            state = torch.randn(1, state_dim)

            # Get raw network outputs
            mean_raw, std_raw = actor.shared_network(state), actor.log_std_head(
                actor.shared_network(state)
            )
            mean_raw = actor.mean_head(actor.shared_network(state))
            std_raw = torch.exp(
                torch.clamp(actor.log_std_head(actor.shared_network(state)), -20, 2)
            )

            # Get final action
            action, _ = actor.get_action_and_log_prob(state)

            print(f"\nRaw mean output: {mean_raw.item():.4f}")
            print(f"Scaled mean: {mean_raw.item() * action_bound:.4f}")
            print(f"Tanh scaled action: {action.item():.4f}")
            print(f"Action within bounds: {abs(action.item()) <= action_bound}")

        return actor
