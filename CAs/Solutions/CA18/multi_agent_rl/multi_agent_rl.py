import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentReplayBuffer:
    """Experience replay buffer for multi-agent systems"""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.observations = np.zeros((capacity, n_agents, obs_dim))
        self.actions = np.zeros((capacity, n_agents, action_dim))
        self.rewards = np.zeros((capacity, n_agents, 1))
        self.next_observations = np.zeros((capacity, n_agents, obs_dim))
        self.dones = np.zeros((capacity, n_agents, 1))

        self.position = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ):
        """Add experience to buffer"""

        self.observations[self.position] = obs
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards.reshape(-1, 1)
        self.next_observations[self.position] = next_obs
        self.dones[self.position] = dones.reshape(-1, 1)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences"""

        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            "observations": torch.FloatTensor(self.observations[indices]).to(device),
            "actions": torch.FloatTensor(self.actions[indices]).to(device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(device),
            "next_observations": torch.FloatTensor(self.next_observations[indices]).to(
                device
            ),
            "dones": torch.FloatTensor(self.dones[indices]).to(device),
        }

    @property
    def __len__(self):
        return self.size


class Actor(nn.Module):
    """Actor network for individual agents"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class Critic(nn.Module):
    """Centralized critic network"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        input_dim = obs_dim + action_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)


class AttentionCritic(nn.Module):
    """Attention-based critic for selective agent focus"""

    def __init__(
        self, obs_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 128
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        self.agent_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(n_agents)
            ]
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        self.critic_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, agent_idx: int
    ) -> torch.Tensor:
        """
        obs: [batch, n_agents, obs_dim]
        actions: [batch, n_agents, action_dim]
        agent_idx: which agent's Q-value to compute
        """

        batch_size = obs.shape[0]

        agent_embeddings = []
        for i in range(self.n_agents):
            agent_obs = obs[:, i]  # [batch, obs_dim]
            agent_action = actions[:, i]  # [batch, action_dim]
            agent_input = torch.cat([agent_obs, agent_action], dim=-1)
            agent_emb = self.agent_encoders[i](agent_input)  # [batch, hidden_dim]
            agent_embeddings.append(agent_emb)

        agent_embeddings = torch.stack(
            agent_embeddings, dim=1
        )  # [batch, n_agents, hidden_dim]

        query = agent_embeddings[:, agent_idx : agent_idx + 1]  # [batch, 1, hidden_dim]
        attended_features, _ = self.attention(query, agent_embeddings, agent_embeddings)

        q_value = self.critic_network(attended_features.squeeze(1))  # [batch, 1]

        return q_value


class CommunicationNetwork(nn.Module):
    """Neural network for inter-agent communication"""

    def __init__(self, obs_dim: int, hidden_dim: int = 64, message_dim: int = 32):
        super().__init__()

        self.obs_dim = obs_dim
        self.message_dim = message_dim

        self.message_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
        )

        self.message_processor = nn.Sequential(
            nn.Linear(obs_dim + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def generate_message(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate message from observation"""
        return self.message_encoder(obs)

    def process_messages(
        self, obs: torch.Tensor, messages: torch.Tensor
    ) -> torch.Tensor:
        """Process received messages with observation"""
        avg_message = messages.mean(dim=1)

        combined = torch.cat([obs, avg_message], dim=-1)

        enhanced_obs = self.message_processor(combined)

        return enhanced_obs


class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient Agent"""

    def __init__(
        self,
        agent_idx: int,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        use_attention: bool = False,
        use_communication: bool = False,
    ):

        self.agent_idx = agent_idx
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.use_attention = use_attention
        self.use_communication = use_communication

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim).to(device)

        if use_attention:
            self.critic = AttentionCritic(obs_dim, action_dim, n_agents).to(device)
            self.critic_target = AttentionCritic(obs_dim, action_dim, n_agents).to(
                device
            )
        else:
            total_obs_dim = obs_dim * n_agents
            total_action_dim = action_dim * n_agents
            self.critic = Critic(total_obs_dim, total_action_dim).to(device)
            self.critic_target = Critic(total_obs_dim, total_action_dim).to(device)

        if use_communication:
            self.comm_network = CommunicationNetwork(obs_dim, message_dim=32).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        if use_communication:
            self.comm_optimizer = torch.optim.Adam(
                self.comm_network.parameters(), lr=lr_actor
            )

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.noise_std = 0.2
        self.noise_decay = 0.995
        self.min_noise = 0.01

    def act(
        self, obs: torch.Tensor, messages: torch.Tensor = None, explore: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action and generate message"""

        if self.use_communication and messages is not None:
            obs = self.comm_network.process_messages(obs, messages)

        action = self.actor(obs)

        if explore:
            noise = torch.randn_like(action) * self.noise_std
            action = torch.clamp(action + noise, -1, 1)

        message = None
        if self.use_communication:
            message = self.comm_network.generate_message(obs)

        return action, message

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        other_actors: List[nn.Module],
        gamma: float = 0.99,
        tau: float = 0.01,
    ):
        """Update actor and critic networks"""

        obs = batch["observations"]  # [batch, n_agents, obs_dim]
        actions = batch["actions"]  # [batch, n_agents, action_dim]
        rewards = batch["rewards"]  # [batch, n_agents, 1]
        next_obs = batch["next_observations"]  # [batch, n_agents, obs_dim]
        dones = batch["dones"]  # [batch, n_agents, 1]

        batch_size = obs.shape[0]

        with torch.no_grad():
            next_actions = torch.zeros_like(actions)
            for i, actor in enumerate(other_actors):
                if i == self.agent_idx:
                    next_actions[:, i] = self.actor_target(next_obs[:, i])
                else:
                    next_actions[:, i] = actor(next_obs[:, i])

            if self.use_attention:
                target_q = self.critic_target(next_obs, next_actions, self.agent_idx)
            else:
                next_obs_flat = next_obs.view(batch_size, -1)
                next_actions_flat = next_actions.view(batch_size, -1)
                target_q = self.critic_target(next_obs_flat, next_actions_flat)

            target_q = (
                rewards[:, self.agent_idx]
                + gamma * (1 - dones[:, self.agent_idx]) * target_q
            )

        if self.use_attention:
            current_q = self.critic(obs, actions, self.agent_idx)
        else:
            obs_flat = obs.view(batch_size, -1)
            actions_flat = actions.view(batch_size, -1)
            current_q = self.critic(obs_flat, actions_flat)

        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        current_actions = actions.clone()
        current_actions[:, self.agent_idx] = self.actor(obs[:, self.agent_idx])

        if self.use_attention:
            actor_loss = -self.critic(obs, current_actions, self.agent_idx).mean()
        else:
            obs_flat = obs.view(batch_size, -1)
            current_actions_flat = current_actions.view(batch_size, -1)
            actor_loss = -self.critic(obs_flat, current_actions_flat).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.soft_update(self.actor_target, self.actor, tau)
        self.soft_update(self.critic_target, self.critic, tau)

        self.noise_std = max(self.noise_std * self.noise_decay, self.min_noise)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": current_q.mean().item(),
            "noise_std": self.noise_std,
        }

    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update target network"""
        target.load_state_dict(source.state_dict())


class MultiAgentEnvironment:
    """Multi-agent environment for testing"""

    def __init__(
        self,
        n_agents: int = 3,
        obs_dim: int = 6,
        action_dim: int = 2,
        env_type: str = "cooperative",
    ):

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_type = env_type
        self.max_steps = 200

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.agent_states = np.random.uniform(-2, 2, (self.n_agents, self.obs_dim))
        self.steps = 0

        return self.get_observations()

    def get_observations(self) -> np.ndarray:
        """Get observations for all agents"""
        observations = np.zeros((self.n_agents, self.obs_dim))

        for i in range(self.n_agents):
            obs = self.agent_states[i].copy()

            obs += np.random.normal(0, 0.1, self.obs_dim)
            observations[i] = obs

        return observations

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Environment step"""
        actions = np.clip(actions, -1, 1)

        for i in range(self.n_agents):
            if self.obs_dim >= 4:
                self.agent_states[i, 2:4] += 0.1 * actions[i, :2]
                self.agent_states[i, 2:4] *= 0.9  # Friction

                self.agent_states[i, :2] += 0.1 * self.agent_states[i, 2:4]
            else:
                self.agent_states[i, :2] += 0.1 * actions[i, :2]

            self.agent_states[i] += np.random.normal(0, 0.02, self.obs_dim)

        rewards = self.compute_rewards()

        self.steps += 1
        dones = np.array([self.steps >= self.max_steps] * self.n_agents)

        for i in range(self.n_agents):
            if np.linalg.norm(self.agent_states[i, :2]) > 5:
                dones[i] = True

        observations = self.get_observations()

        return observations, rewards, dones, {}

    def compute_rewards(self) -> np.ndarray:
        """Compute rewards based on environment type"""
        rewards = np.zeros(self.n_agents)

        if self.env_type == "cooperative":
            center = np.mean(self.agent_states[:, :2], axis=0)

            for i in range(self.n_agents):
                center_reward = -np.linalg.norm(self.agent_states[i, :2])

                cohesion_reward = 0
                for j in range(self.n_agents):
                    if i != j:
                        dist = np.linalg.norm(
                            self.agent_states[i, :2] - self.agent_states[j, :2]
                        )
                        cohesion_reward += -0.1 * dist

                rewards[i] = center_reward + 0.5 * cohesion_reward

        elif self.env_type == "competitive":
            target = np.array([0, 0])  # Shared resource at origin

            distances = [
                np.linalg.norm(self.agent_states[i, :2] - target)
                for i in range(self.n_agents)
            ]
            closest_agent = np.argmin(distances)

            for i in range(self.n_agents):
                if i == closest_agent:
                    rewards[i] = 1.0  # Winner gets reward
                else:
                    rewards[i] = -0.1  # Others get penalty

        elif self.env_type == "mixed":
            team_size = self.n_agents // 2

            for i in range(self.n_agents):
                team_id = i // team_size

                team_reward = 0
                for j in range(self.n_agents):
                    if j // team_size == team_id and i != j:
                        dist = np.linalg.norm(
                            self.agent_states[i, :2] - self.agent_states[j, :2]
                        )
                        team_reward += -0.1 * dist

                comp_reward = 0
                for j in range(self.n_agents):
                    if j // team_size != team_id:
                        dist = np.linalg.norm(
                            self.agent_states[i, :2] - self.agent_states[j, :2]
                        )
                        comp_reward += 0.05 * max(
                            0, 2 - dist
                        )  # Reward for keeping distance

                rewards[i] = team_reward + comp_reward

        return rewards
