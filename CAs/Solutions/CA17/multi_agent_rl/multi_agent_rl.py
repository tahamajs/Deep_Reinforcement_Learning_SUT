import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx

# Multi-Agent Experience Replay Buffer
class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent systems"""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Storage
        self.observations = np.zeros((capacity, n_agents, obs_dim))
        self.actions = np.zeros((capacity, n_agents, action_dim))
        self.rewards = np.zeros((capacity, n_agents))
        self.next_observations = np.zeros((capacity, n_agents, obs_dim))
        self.dones = np.zeros((capacity, n_agents), dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_obs: np.ndarray, dones: np.ndarray):
        """Add experience to buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer"""
        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_observations': torch.FloatTensor(self.next_observations[indices]),
            'dones': torch.BoolTensor(self.dones[indices])
        }

    def __len__(self):
        return self.size


# Actor Network for MADDPG
class MADDPGActor(nn.Module):
    """Actor network for MADDPG - decentralized policy"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Assume actions are bounded [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


# Critic Network for MADDPG
class MADDPGCritic(nn.Module):
    """Critic network for MADDPG - centralized value function"""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 128):
        super().__init__()

        # Input: all agents' observations and actions
        input_dim = (obs_dim + action_dim) * n_agents

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            actions: [batch, n_agents, action_dim]
        Returns:
            Q-values: [batch, 1]
        """
        # Flatten observations and actions
        obs_flat = obs.reshape(obs.shape[0], -1)
        actions_flat = actions.reshape(actions.shape[0], -1)

        # Concatenate all information
        inputs = torch.cat([obs_flat, actions_flat], dim=1)

        return self.network(inputs)


# MADDPG Agent
class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient Agent"""

    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, n_agents: int,
                 lr_actor: float = 1e-3, lr_critic: float = 1e-3, gamma: float = 0.99,
                 tau: float = 0.01, noise_std: float = 0.1):

        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

        # Networks
        self.actor = MADDPGActor(obs_dim, action_dim)
        self.critic = MADDPGCritic(obs_dim, action_dim, n_agents)
        self.target_actor = MADDPGActor(obs_dim, action_dim)
        self.target_critic = MADDPGCritic(obs_dim, action_dim, n_agents)

        # Copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Exploration noise
        self.noise = Normal(0, noise_std)

    def act(self, obs: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Select action given observation"""
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs)
            if add_noise:
                noise = self.noise.sample(action.shape)
                action = torch.clamp(action + noise, -1, 1)
        self.actor.train()
        return action

    def update_critic(self, batch: Dict[str, torch.Tensor],
                     target_actions: torch.Tensor) -> float:
        """Update critic network"""
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards'][:, self.agent_id].unsqueeze(1)
        next_obs = batch['next_observations']
        dones = batch['dones'][:, self.agent_id].unsqueeze(1)

        # Current Q-values
        current_q = self.critic(obs, actions)

        # Target Q-values
        with torch.no_grad():
            target_q = self.target_critic(next_obs, target_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones.float())

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, batch: Dict[str, torch.Tensor],
                    agent_actions: List[torch.Tensor]) -> float:
        """Update actor network"""
        obs = batch['observations']

        # Construct joint action with current agent's policy
        actions = torch.stack(agent_actions, dim=1)  # [batch, n_agents, action_dim]
        actions[:, self.agent_id] = self.actor(obs[:, self.agent_id])

        # Actor loss (negative Q-value)
        actor_loss = -self.critic(obs, actions).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return actor_loss.item()

    def soft_update(self):
        """Soft update of target networks"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# Communication Network
class CommunicationNetwork(nn.Module):
    """Neural communication network for multi-agent coordination"""

    def __init__(self, obs_dim: int, comm_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.obs_dim = obs_dim
        self.comm_dim = comm_dim

        # Message generation network
        self.msg_generator = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, comm_dim),
            nn.Tanh()
        )

        # Message processing network
        self.msg_processor = nn.Sequential(
            nn.Linear(obs_dim + comm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def generate_message(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate message from observation"""
        return self.msg_generator(obs)

    def process_messages(self, obs: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Process received messages with observation"""
        # Average received messages (could use attention instead)
        avg_message = messages.mean(dim=1)  # Average over senders

        # Combine with observation
        combined = torch.cat([obs, avg_message], dim=-1)

        return self.msg_processor(combined)


# Communicative Multi-Agent System
class CommMADDPG(nn.Module):
    """MADDPG with learned communication"""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int,
                 comm_dim: int = 16, hidden_dim: int = 128):
        super().__init__()

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.comm_dim = comm_dim

        # Communication networks for each agent
        self.comm_nets = nn.ModuleList([
            CommunicationNetwork(obs_dim, comm_dim, hidden_dim)
            for _ in range(n_agents)
        ])

        # Actor networks with communication input
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            ) for _ in range(n_agents)
        ])

        # Centralized critic
        total_input_dim = (obs_dim + action_dim) * n_agents + comm_dim * n_agents
        self.critic = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observations: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with communication

        Args:
            observations: [batch, n_agents, obs_dim]
            training: Whether in training mode

        Returns:
            Dictionary with actions, messages, and processed features
        """
        batch_size = observations.shape[0]

        # Generate messages for each agent
        messages = []
        for i in range(self.n_agents):
            msg = self.comm_nets[i].generate_message(observations[:, i])
            messages.append(msg)
        messages = torch.stack(messages, dim=1)  # [batch, n_agents, comm_dim]

        # Process messages and observations
        processed_features = []
        actions = []

        for i in range(self.n_agents):
            # Get messages from other agents
            other_messages = torch.cat([messages[:, :i], messages[:, i+1:]], dim=1)

            # Process observation with received messages
            features = self.comm_nets[i].process_messages(observations[:, i], other_messages)
            processed_features.append(features)

            # Generate action
            action = self.actors[i](features)
            actions.append(action)

        actions = torch.stack(actions, dim=1)  # [batch, n_agents, action_dim]
        processed_features = torch.stack(processed_features, dim=1)

        return {
            'actions': actions,
            'messages': messages,
            'features': processed_features
        }


# Multi-Agent Environment (Predator-Prey)
class PredatorPreyEnvironment:
    """Multi-agent predator-prey environment"""

    def __init__(self, n_predators: int = 2, n_prey: int = 1, grid_size: int = 10,
                 max_steps: int = 100):
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.n_agents = n_predators + n_prey
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Agent positions
        self.predator_positions = []
        self.prey_positions = []

        # Environment state
        self.step_count = 0
        self.done = False

        # Action mapping (0: up, 1: down, 2: left, 3: right, 4: stay)
        self.action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay
        }

        self.observation_dim = 4 + 2 * (n_predators + n_prey - 1)  # position + relative positions
        self.action_dim = 5

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.step_count = 0
        self.done = False

        # Random initial positions
        self.predator_positions = []
        for _ in range(self.n_predators):
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.predator_positions.append(pos)

        self.prey_positions = []
        for _ in range(self.n_prey):
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.prey_positions.append(pos)

        return self._get_observations()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """Take environment step"""
        self.step_count += 1

        # Move predators
        for i, action in enumerate(actions[:self.n_predators]):
            dx, dy = self.action_map[action]
            x, y = self.predator_positions[i]
            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)
            self.predator_positions[i] = (new_x, new_y)

        # Move prey (simple random policy)
        for i in range(self.n_prey):
            action = np.random.randint(5)
            dx, dy = self.action_map[action]
            x, y = self.prey_positions[i]
            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)
            self.prey_positions[i] = (new_x, new_y)

        # Calculate rewards
        rewards = self._calculate_rewards()

        # Check termination
        self.done = (self.step_count >= self.max_steps or
                    self._check_capture())

        observations = self._get_observations()

        return observations, rewards, self.done, {}

    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents"""
        observations = []

        # Predator observations
        for i in range(self.n_predators):
            obs = self._get_agent_observation(i, is_predator=True)
            observations.append(obs)

        # Prey observations
        for i in range(self.n_prey):
            obs = self._get_agent_observation(i, is_predator=False)
            observations.append(obs)

        return np.array(observations)

    def _get_agent_observation(self, agent_idx: int, is_predator: bool) -> np.ndarray:
        """Get observation for single agent"""
        if is_predator:
            agent_pos = self.predator_positions[agent_idx]
            other_predators = [pos for i, pos in enumerate(self.predator_positions) if i != agent_idx]
            other_agents = other_predators + self.prey_positions
        else:
            agent_pos = self.prey_positions[agent_idx]
            other_prey = [pos for i, pos in enumerate(self.prey_positions) if i != agent_idx]
            other_agents = self.predator_positions + other_prey

        # Agent's position (normalized)
        obs = [agent_pos[0] / self.grid_size, agent_pos[1] / self.grid_size]

        # Agent's velocity (placeholder - could track from previous positions)
        obs.extend([0.0, 0.0])

        # Relative positions of other agents
        for other_pos in other_agents:
            rel_x = (other_pos[0] - agent_pos[0]) / self.grid_size
            rel_y = (other_pos[1] - agent_pos[1]) / self.grid_size
            obs.extend([rel_x, rel_y])

        # Pad observation if needed
        while len(obs) < self.observation_dim:
            obs.append(0.0)

        return np.array(obs[:self.observation_dim])

    def _calculate_rewards(self) -> np.ndarray:
        """Calculate rewards for all agents"""
        rewards = np.zeros(self.n_agents)

        # Predator rewards
        for i in range(self.n_predators):
            pred_pos = self.predator_positions[i]

            # Reward for being close to prey
            min_distance = float('inf')
            for prey_pos in self.prey_positions:
                distance = abs(pred_pos[0] - prey_pos[0]) + abs(pred_pos[1] - prey_pos[1])
                min_distance = min(min_distance, distance)

            rewards[i] = 1.0 / (min_distance + 1)  # Closer = higher reward

            # Bonus for capture
            if self._check_capture():
                rewards[i] += 10.0

        # Prey rewards (negative of average predator reward)
        prey_reward = -np.mean(rewards[:self.n_predators])
        for i in range(self.n_predators, self.n_agents):
            rewards[i] = prey_reward

        return rewards

    def _check_capture(self) -> bool:
        """Check if any prey is captured"""
        for prey_pos in self.prey_positions:
            for pred_pos in self.predator_positions:
                if pred_pos == prey_pos:
                    return True
        return False

    def render(self):
        """Render environment"""
        grid = np.zeros((self.grid_size, self.grid_size))

        # Mark predators as 1
        for pos in self.predator_positions:
            grid[pos] = 1

        # Mark prey as 2
        for pos in self.prey_positions:
            grid[pos] = 2

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='viridis')
        plt.colorbar(label='Agent Type (0: Empty, 1: Predator, 2: Prey)')
        plt.title(f'Predator-Prey Environment (Step: {self.step_count})')
        plt.show()