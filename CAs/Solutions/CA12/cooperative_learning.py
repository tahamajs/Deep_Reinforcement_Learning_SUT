import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from setup import device, ma_config


class Actor(nn.Module):
    """Actor network for MADDPG."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.max_action * self.net(obs)


class Critic(nn.Module):
    """Centralized critic for MADDPG."""

    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=128):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, actions):
        return self.net(torch.cat([obs, actions], dim=-1))


class MADDPGAgent:
    """Single agent in MADDPG framework."""

    def __init__(
        self,
        agent_id,
        obs_dim,
        action_dim,
        total_obs_dim,
        total_action_dim,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
    ):
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(total_obs_dim, total_action_dim).to(device)
        self.target_actor = Actor(obs_dim, action_dim).to(device)
        self.target_critic = Critic(total_obs_dim, total_action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.noise_scale = 0.1
        self.noise_decay = 0.9999

    def act(self, obs, add_noise=True):
        """Select action given observation."""
        obs = torch.FloatTensor(obs).to(device)
        action = self.actor(obs).cpu().data.numpy()

        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action += noise
            self.noise_scale *= self.noise_decay

        return np.clip(action, -1, 1)

    def update_critic(self, obs, actions, rewards, next_obs, next_actions, dones):
        """Update critic network."""
        obs = torch.FloatTensor(obs).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        next_actions = torch.FloatTensor(next_actions).to(device)
        dones = torch.BoolTensor(dones).to(device)

        current_q = self.critic(obs, actions).squeeze()

        with torch.no_grad():
            target_q = self.target_critic(next_obs, next_actions).squeeze()
            target_q = rewards + self.gamma * target_q * ~dones

        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, obs, actions):
        """Update actor network."""
        obs = torch.FloatTensor(obs).to(device)
        actions = torch.FloatTensor(actions).to(device)

        actions_pred = actions.clone()
        agent_obs = obs[:, self.agent_id]
        actions_pred[:, self.agent_id] = self.actor(agent_obs)

        actor_loss = -self.critic(
            obs.view(obs.size(0), -1), actions_pred.view(actions_pred.size(0), -1)
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return actor_loss.item()

    def soft_update(self):
        """Soft update of target networks."""
        for target, source in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target.data.copy_(self.tau * source.data + (1.0 - self.tau) * target.data)

        for target, source in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target.data.copy_(self.tau * source.data + (1.0 - self.tau) * target.data)


class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient."""

    def __init__(self, n_agents, obs_dim, action_dim, buffer_size=100000):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        total_obs_dim = n_agents * obs_dim
        total_action_dim = n_agents * action_dim

        self.agents = [
            MADDPGAgent(i, obs_dim, action_dim, total_obs_dim, total_action_dim)
            for i in range(n_agents)
        ]

        self.replay_buffer = ReplayBuffer(buffer_size)

    def act(self, observations, add_noise=True):
        """Get actions from all agents."""
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(observations[i], add_noise)
            actions.append(action)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Store experience and update agents."""

        self.replay_buffer.push(states, actions, rewards, next_states, dones)

        if len(self.replay_buffer) > ma_config.batch_size:
            self.update()

    def update(self):
        """Update all agents."""
        batch = self.replay_buffer.sample(ma_config.batch_size)
        states, actions, rewards, next_states, dones = batch

        states_flat = np.array(states).reshape(len(states), -1)
        actions_flat = np.array(actions).reshape(len(actions), -1)
        next_states_flat = np.array(next_states).reshape(len(next_states), -1)

        next_actions = []
        for i, agent in enumerate(self.agents):
            next_obs = torch.FloatTensor(next_states).to(device)[:, i]
            next_action = agent.target_actor(next_obs)
            next_actions.append(next_action)

        next_actions_flat = torch.cat(next_actions, dim=-1).cpu().data.numpy()

        losses = {"actor": [], "critic": []}
        for i, agent in enumerate(self.agents):
            agent_rewards = np.array(rewards)[:, i]
            agent_dones = np.array(dones)

            critic_loss = agent.update_critic(
                states_flat,
                actions_flat,
                agent_rewards,
                next_states_flat,
                next_actions_flat,
                agent_dones,
            )
            losses["critic"].append(critic_loss)

            actor_loss = agent.update_actor(states, actions)
            losses["actor"].append(actor_loss)

            agent.soft_update()

        return losses


class ReplayBuffer:
    """Replay buffer for multi-agent experiences."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states, dones):
        """Store a transition."""
        self.buffer.append((states, actions, rewards, next_states, dones))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class VDNAgent(nn.Module):
    """Individual agent network for VDN."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(VDNAgent, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class VDN:
    """Value Decomposition Network for cooperative MARL."""

    def __init__(self, n_agents, obs_dim, action_dim, lr=1e-3):
        self.n_agents = n_agents
        self.agents = [
            VDNAgent(obs_dim, action_dim).to(device) for _ in range(n_agents)
        ]
        self.target_agents = [
            VDNAgent(obs_dim, action_dim).to(device) for _ in range(n_agents)
        ]

        for agent, target in zip(self.agents, self.target_agents):
            target.load_state_dict(agent.state_dict())

        self.optimizers = [
            optim.Adam(agent.parameters(), lr=lr) for agent in self.agents
        ]
        self.replay_buffer = ReplayBuffer(10000)

    def act(self, observations, epsilon=0.1):
        """Epsilon-greedy action selection."""
        actions = []
        for i, agent in enumerate(self.agents):
            if np.random.random() < epsilon:
                action = np.random.randint(agent.net[-1].out_features)
            else:
                obs = torch.FloatTensor(observations[i]).to(device)
                q_values = agent(obs)
                action = q_values.argmax().item()
            actions.append(action)
        return actions

    def update(self, batch_size=32):
        """Update VDN agents."""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        total_loss = 0

        team_rewards = torch.FloatTensor([sum(r) for r in rewards]).to(device)
        team_dones = torch.BoolTensor([any(d) for d in dones]).to(device)

        for i, (agent, target_agent, optimizer) in enumerate(
            zip(self.agents, self.target_agents, self.optimizers)
        ):
            agent_states = torch.FloatTensor([s[i] for s in states]).to(device)
            agent_actions = torch.LongTensor([a[i] for a in actions]).to(device)
            agent_next_states = torch.FloatTensor([s[i] for s in next_states]).to(
                device
            )

            q_values = agent(agent_states)
            q_values = q_values.gather(1, agent_actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q_values = target_agent(agent_next_states).max(1)[0]
                target_q = team_rewards + 0.99 * next_q_values * ~team_dones

            loss = F.mse_loss(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        tau = 0.005
        for agent, target_agent in zip(self.agents, self.target_agents):
            for param, target_param in zip(
                agent.parameters(), target_agent.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

        return total_loss / self.n_agents


print("🤖 Cooperative multi-agent algorithms implemented successfully!")
print("✅ MADDPG, VDN, and supporting utilities ready for training!")
