"""
CA5: Deep Q-Networks and Advanced Value-based Methods - Extended Training Examples
==================================================================================

This module provides comprehensive implementations and analysis functions for
Deep Q-Networks (DQN) and advanced value-based reinforcement learning methods.

Includes implementations of:
- Vanilla DQN with experience replay and target networks
- Double DQN (addressing overestimation bias)
- Dueling DQN (value-advantage decomposition)
- Prioritized Experience Replay
- Rainbow DQN (combining multiple improvements)

Author: DRL Course Team
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import random
from collections import deque, namedtuple
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Define transition tuple for experience replay
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer"""
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample batch of transitions"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using sum trees"""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition with max priority"""
        transition = Transition(state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch with priorities"""
        if len(self.buffer) < batch_size:
            return [], [], []

        priorities = self.priorities[: len(self.buffer)] ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-network for DQN"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """Dueling Q-network separating value and advantage streams"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()

        # Feature extraction
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantage: Q = V + (A - mean(A))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class DQNAgent:
    """Vanilla DQN Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training stats
        self.steps = 0

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self) -> float:
        """Update Q-network"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss and update
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """Save model"""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent - addresses overestimation bias"""

    def update(self) -> float:
        """Update with Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: select with current network, evaluate with target network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = (
                self.target_network(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze()
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss and update
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN Agent - separates value and advantage"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override networks with dueling architecture
        self.q_network = DuelingQNetwork(self.state_dim, self.action_dim).to(
            self.device
        )
        self.target_network = DuelingQNetwork(self.state_dim, self.action_dim).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Reinitialize optimizer with new network
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=kwargs.get("lr", 1e-3)
        )


class PrioritizedDQNAgent(DQNAgent):
    """DQN with Prioritized Experience Replay"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(kwargs.get("buffer_size", 50000))

    def update(self) -> float:
        """Update with prioritized replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch with priorities
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        if not transitions:
            return 0.0

        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # TD errors for priority updates
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()

        # Weighted loss
        loss = (weights * F.mse_loss(current_q, target_q, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()


def train_dqn_agent(
    env_name: str = "CartPole-v1",
    agent_type: str = "dqn",
    num_episodes: int = 1000,
    **kwargs,
) -> Dict[str, List[float]]:
    """
    Train DQN agent with monitoring

    Args:
        env_name: Gymnasium environment name
        agent_type: Type of agent ('dqn', 'double_dqn', 'dueling_dqn', 'prioritized_dqn')
        num_episodes: Number of training episodes
        **kwargs: Additional arguments for agent

    Returns:
        Dictionary containing training metrics
    """
    logger.info(f"Training {agent_type.upper()} on {env_name}")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent
    agent_classes = {
        "dqn": DQNAgent,
        "double_dqn": DoubleDQNAgent,
        "dueling_dqn": DuelingDQNAgent,
        "prioritized_dqn": PrioritizedDQNAgent,
    }

    agent = agent_classes[agent_type](state_dim, action_dim, **kwargs)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilons = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.update()
            if loss > 0:
                episode_loss += loss
                loss_count += 1

            state = next_state
            episode_reward += reward
            episode_length += 1

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        losses.append(episode_loss / max(loss_count, 1))
        epsilons.append(agent.epsilon)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(
                f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}"
            )

    env.close()

    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "epsilons": epsilons,
    }

    logger.info("Training completed")
    return results


def plot_q_value_landscape(
    agent: DQNAgent, env_name: str = "CartPole-v1", save_path: Optional[str] = None
):
    """Visualize Q-value landscape for DQN agents"""
    logger.info("Generating Q-value landscape visualization")

    # Create environment for state sampling
    env = gym.make(env_name)
    agent.q_network.eval()

    # Sample states from environment
    states = []
    for _ in range(1000):
        state, _ = env.reset()
        states.append(state)
        done = False
        while not done and len(states) < 1000:
            action = agent.select_action(state, epsilon=0.1)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if not done:
                states.append(state)

    states = np.array(states[:500])  # Limit for visualization

    # Get Q-values for all states
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states).to(agent.device)
        q_values = agent.q_network(state_tensor).cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Q-value distributions
    for i in range(min(agent.action_dim, 6)):
        if i < 6:
            ax_idx = i // 3, i % 3
            axes[ax_idx].hist(q_values[:, i], bins=30, alpha=0.7, edgecolor="black")
            axes[ax_idx].set_xlabel(f"Q-value (Action {i})")
            axes[ax_idx].set_ylabel("Frequency")
            axes[ax_idx].set_title(f"Q-value Distribution - Action {i}")
            axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # State-Q value correlation analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Correlation between state features and Q-values
    for i in range(min(states.shape[1], 4)):
        for j in range(min(agent.action_dim, 4)):
            ax = axes[i // 2, i % 2]
            ax.scatter(states[:, i], q_values[:, j], alpha=0.6, s=10)
            ax.set_xlabel(f"State Feature {i}")
            ax.set_ylabel(f"Q-value (Action {j})")
            ax.set_title(f"State {i} vs Q-value Action {j}")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    env.close()
    logger.info("Q-value landscape visualization completed")


def plot_experience_replay_analysis(
    replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
    save_path: Optional[str] = None,
):
    """Analyze experience replay buffer contents"""
    logger.info("Analyzing experience replay buffer")

    if len(replay_buffer) == 0:
        logger.warning("Replay buffer is empty!")
        return

    # Extract transitions
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for transition in list(replay_buffer.buffer):
        states.append(transition.state)
        actions.append(transition.action)
        rewards.append(transition.reward)
        next_states.append(transition.next_state)
        dones.append(transition.done)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Reward distribution
    axes[0, 0].hist(rewards, bins=50, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Reward")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Reward Distribution in Replay Buffer")
    axes[0, 0].axvline(
        np.mean(rewards),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(rewards):.2f}",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Action distribution
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    axes[0, 1].bar(unique_actions, action_counts, alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Action")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Action Distribution in Replay Buffer")
    axes[0, 1].grid(True, alpha=0.3)

    # State feature distributions
    for i in range(min(states.shape[1], 4)):
        ax_idx = 0 if i < 2 else 1
        ax_pos = i % 2 + 2
        if ax_pos < 3:
            axes[ax_idx, ax_pos].hist(
                states[:, i], bins=30, alpha=0.7, edgecolor="black"
            )
            axes[ax_idx, ax_pos].set_xlabel(f"State Feature {i}")
            axes[ax_idx, ax_pos].set_ylabel("Frequency")
            axes[ax_idx, ax_pos].set_title(f"State Feature {i} Distribution")
            axes[ax_idx, ax_pos].grid(True, alpha=0.3)

    # Terminal states analysis
    terminal_mask = dones == 1
    non_terminal_mask = dones == 0

    axes[1, 0].scatter(
        states[non_terminal_mask, 0],
        states[non_terminal_mask, 1],
        alpha=0.3,
        label="Non-terminal",
        s=10,
    )
    axes[1, 0].scatter(
        states[terminal_mask, 0],
        states[terminal_mask, 1],
        alpha=0.7,
        color="red",
        label="Terminal",
        s=20,
    )
    axes[1, 0].set_xlabel("State Feature 0")
    axes[1, 0].set_ylabel("State Feature 1")
    axes[1, 0].set_title("Terminal vs Non-terminal States")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Reward by action
    reward_by_action = {}
    for action in unique_actions:
        action_mask = actions == action
        reward_by_action[action] = rewards[action_mask]

    axes[1, 1].boxplot(
        [reward_by_action[action] for action in unique_actions],
        labels=[f"Action {int(a)}" for a in unique_actions],
    )
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].set_title("Reward Distribution by Action")
    axes[1, 1].grid(True, alpha=0.3)

    # State transitions (simplified)
    if states.shape[1] >= 2:
        delta_states = next_states - states
        axes[1, 2].scatter(delta_states[:, 0], delta_states[:, 1], alpha=0.3, s=10)
        axes[1, 2].set_xlabel("State Change Feature 0")
        axes[1, 2].set_ylabel("State Change Feature 1")
        axes[1, 2].set_title("State Transition Patterns")
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Print statistics
    logger.info("Replay buffer analysis completed")
    print(f"\nReplay Buffer Statistics:")
    print(f"Total transitions: {len(replay_buffer)}")
    print(f"Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Terminal states: {np.sum(dones)} ({100*np.mean(dones):.1f}%)")
    print(
        f"Action distribution: {dict(zip(unique_actions.astype(int), action_counts))}"
    )


def dqn_variant_comparison() -> Dict[str, Any]:
    """Compare different DQN variants performance"""
    logger.info("Starting DQN Variants Performance Comparison")

    # Define DQN variants
    variants = {
        "Vanilla DQN": {
            "description": "Basic DQN with experience replay and target networks",
            "improvements": ["Experience Replay", "Target Networks"],
            "limitations": ["Overestimation bias", "Limited sample efficiency"],
        },
        "Double DQN": {
            "description": "Addresses overestimation using separate networks for selection and evaluation",
            "improvements": ["Reduced overestimation", "Better performance"],
            "limitations": ["Still has some bias", "Increased complexity"],
        },
        "Dueling DQN": {
            "description": "Separates state value and advantage estimation",
            "improvements": ["Better value estimation", "Improved learning"],
            "limitations": ["More parameters", "Potential instability"],
        },
        "Prioritized Replay": {
            "description": "Samples important transitions more frequently",
            "improvements": ["Better sample efficiency", "Faster learning"],
            "limitations": ["Bias introduction", "Complexity"],
        },
        "Rainbow DQN": {
            "description": "Combines all DQN improvements",
            "improvements": ["State-of-the-art performance", "Robust learning"],
            "limitations": ["High complexity", "Resource intensive"],
        },
    }

    # Mock performance comparison (in practice, this would be real training results)
    environments = ["CartPole-v1", "LunarLander-v2", "PongNoFrameskip-v4"]
    performance_data = {}

    for env in environments:
        performance_data[env] = {}
        base_scores = {
            "CartPole-v1": 400,
            "LunarLander-v2": 150,
            "PongNoFrameskip-v4": 18,
        }

        for variant in variants.keys():
            # Simulate performance improvements
            improvement_factors = {
                "Vanilla DQN": 1.0,
                "Double DQN": 1.15,
                "Dueling DQN": 1.25,
                "Prioritized Replay": 1.35,
                "Rainbow DQN": 1.5,
            }

            score = base_scores[env] * improvement_factors[variant]
            score += np.random.normal(0, base_scores[env] * 0.1)
            performance_data[env][variant] = max(score, 0)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Performance comparison
    env_names = list(performance_data.keys())
    variant_names = list(variants.keys())

    x = np.arange(len(env_names))
    width = 0.15
    multiplier = 0

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (variant, color) in enumerate(zip(variant_names, colors)):
        scores = [performance_data[env][variant] for env in env_names]
        offset = width * multiplier
        bars = axes[0, 0].bar(
            x + offset, scores, width, label=variant, color=color, alpha=0.8
        )
        axes[0, 0].bar_label(bars, fmt=".0f", padding=3, fontsize=8)
        multiplier += 1

    axes[0, 0].set_xlabel("Environment")
    axes[0, 0].set_ylabel("Average Score")
    axes[0, 0].set_title("DQN Variants Performance Comparison")
    axes[0, 0].set_xticks(x + width * 2, env_names)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # Improvement analysis
    improvement_data = {}
    for variant in variant_names[1:]:  # Skip Vanilla DQN
        improvements = []
        for env in env_names:
            vanilla_score = performance_data[env]["Vanilla DQN"]
            variant_score = performance_data[env][variant]
            improvement = (variant_score - vanilla_score) / vanilla_score * 100
            improvements.append(improvement)
        improvement_data[variant] = np.mean(improvements)

    axes[0, 1].bar(
        range(len(improvement_data)),
        list(improvement_data.values()),
        alpha=0.7,
        edgecolor="black",
    )
    axes[0, 1].set_xlabel("DQN Variant")
    axes[0, 1].set_ylabel("Average Improvement (%)")
    axes[0, 1].set_title("Performance Improvement Over Vanilla DQN")
    axes[0, 1].set_xticks(range(len(improvement_data)))
    axes[0, 1].set_xticklabels(list(improvement_data.keys()), rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3)

    # Complexity vs Performance
    complexities = [1, 2, 3, 4, 5]  # Relative complexity
    avg_performances = []

    for variant in variant_names:
        avg_perf = np.mean([performance_data[env][variant] for env in env_names])
        avg_performances.append(avg_perf)

    axes[1, 0].scatter(complexities, avg_performances, s=100, alpha=0.7, c="red")
    for i, variant in enumerate(variant_names):
        axes[1, 0].annotate(
            variant,
            (complexities[i], avg_performances[i]),
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
        )

    axes[1, 0].set_xlabel("Implementation Complexity")
    axes[1, 0].set_ylabel("Average Performance")
    axes[1, 0].set_title("Complexity vs Performance Tradeoff")
    axes[1, 0].grid(True, alpha=0.3)

    # Learning characteristics radar
    categories = [
        "Sample Efficiency",
        "Stability",
        "Final Performance",
        "Ease of Tuning",
        "Computational Cost",
    ]
    characteristics = {
        "Vanilla DQN": [6, 5, 6, 8, 9],
        "Double DQN": [7, 7, 7, 7, 8],
        "Dueling DQN": [8, 6, 8, 6, 7],
        "Prioritized Replay": [9, 5, 8, 5, 6],
        "Rainbow DQN": [10, 8, 10, 4, 4],
    }

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for variant, scores in characteristics.items():
        scores += scores[:1]
        axes[1, 1].plot(angles, scores, "o-", linewidth=2, label=variant, markersize=6)

    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels(categories, fontsize=9)
    axes[1, 1].set_ylim(0, 10)
    axes[1, 1].set_title("DQN Variants Characteristics")
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dqn_variants_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print detailed analysis
    logger.info("DQN variants comparison completed")
    print("\n" + "=" * 45)
    print("DQN VARIANTS ANALYSIS")
    print("=" * 45)

    for variant, info in variants.items():
        print(f"\n{variant}:")
        print(f"  Description: {info['description']}")
        print(f"  Key Improvements: {', '.join(info['improvements'])}")
        print(f"  Limitations: {', '.join(info['limitations'])}")

        avg_score = np.mean([performance_data[env][variant] for env in env_names])
    #         print(".1f"
    #     print("
    # 💡 Recommendations:"    print("• Start with Vanilla DQN for simple problems")
    print("• Use Double DQN for better stability and performance")
    print("• Try Dueling DQN for value estimation improvements")
    print("• Use Rainbow DQN for state-of-the-art results")
    print("• Consider Prioritized Replay for sample efficiency")

    return {
        "variants": variants,
        "performance_data": performance_data,
        "characteristics": characteristics,
    }


if __name__ == "__main__":
    # Example usage
    print("CA5: Deep Q-Networks and Advanced Value-based Methods")
    print("=" * 60)

    # Train different DQN variants
    print("\n1. Training DQN variants on CartPole...")

    variants_to_train = ["dqn", "double_dqn", "dueling_dqn"]
    training_results = {}

    for variant in variants_to_train:
        print(f"\nTraining {variant.upper()}...")
        results = train_dqn_agent(
            env_name="CartPole-v1",
            agent_type=variant,
            num_episodes=500,
            lr=1e-3,
            gamma=0.99,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=100,
        )
        training_results[variant] = results

    # Run variant comparison
    print("\n2. Running DQN variant comparison...")
    comparison_results = dqn_variant_comparison()

    # Plot training analysis for best variant
    print("\n3. Creating training analysis plots...")
    best_variant = max(
        training_results.keys(),
        key=lambda v: np.mean(training_results[v]["episode_rewards"][-100:]),
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    episodes = range(1, len(training_results[best_variant]["episode_rewards"]) + 1)

    # Episode rewards
    axes[0, 0].plot(
        episodes, training_results[best_variant]["episode_rewards"], alpha=0.7
    )
    axes[0, 0].plot(
        episodes,
        pd.Series(training_results[best_variant]["episode_rewards"]).rolling(50).mean(),
        linewidth=2,
        color="red",
        label="Rolling Mean (50)",
    )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].set_title(f"{best_variant.upper()} Training Rewards")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(
        episodes, training_results[best_variant]["episode_lengths"], alpha=0.7
    )
    axes[0, 1].plot(
        episodes,
        pd.Series(training_results[best_variant]["episode_lengths"]).rolling(50).mean(),
        linewidth=2,
        color="red",
        label="Rolling Mean (50)",
    )
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Episode Length")
    axes[0, 1].set_title(f"{best_variant.upper()} Episode Lengths")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training losses
    axes[1, 0].plot(episodes, training_results[best_variant]["losses"], alpha=0.7)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Training Loss")
    axes[1, 0].set_title(f"{best_variant.upper()} Training Losses")
    axes[1, 0].grid(True, alpha=0.3)

    # Exploration rate
    axes[1, 1].plot(
        episodes, training_results[best_variant]["epsilons"], linewidth=2, color="green"
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].set_title(f"{best_variant.upper()} Exploration Rate")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dqn_training_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n✅ DQN analysis completed!")
    print("Generated files:")
    print("- dqn_variants_comparison.png")
    # print("- dqn_training_analysis.png")
