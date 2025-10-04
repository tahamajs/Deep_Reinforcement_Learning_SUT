import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class ModelFreeAgent:
    """Base class for model-free RL agents"""

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, state, epsilon=0.1):
        """Select action using epsilon-greedy policy"""
        raise NotImplementedError

    def update(self, batch_size=32):
        """Update agent parameters"""
        raise NotImplementedError


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent(ModelFreeAgent):
    """Deep Q-Network agent with experience replay and target network"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        buffer_size=10000,
        target_update_freq=100,
    ):
        super().__init__(state_dim, action_dim, learning_rate, gamma)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        from ..buffers.replay_buffer import ReplayBuffer

        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training statistics
        self.training_step = 0
        self.epsilon = epsilon_start

    def act(self, state, epsilon=None):
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self, batch_size=32):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.BoolTensor(batch["dones"]).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * self.training_step
            / self.epsilon_decay,
        )

        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_step += 1

        return loss.item()


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def __len__(self):
        return len(self.buffer)


class MultiAgentDQN:
    """Independent DQN agents with optional communication"""

    def __init__(
        self,
        n_agents,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=1e-3,
        enable_communication=False,
    ):
        self.n_agents = n_agents
        self.agents = []

        for i in range(n_agents):
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                learning_rate=lr,
            )
            self.agents.append(agent)

        self.enable_communication = enable_communication
        if enable_communication:
            # Communication network
            self.comm_net = nn.Sequential(
                nn.Linear(state_dim * n_agents, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_agents * 8),  # 8-dim message per agent
                nn.Tanh(),
            )
            self.comm_optimizer = optim.Adam(self.comm_net.parameters(), lr=lr)

    def act(self, observations, epsilon=0.1):
        """Select actions for all agents"""
        actions = []

        if self.enable_communication:
            # Generate communication messages
            all_obs = torch.cat([torch.FloatTensor(obs) for obs in observations])
            with torch.no_grad():  # No gradients needed for inference
                messages = self.comm_net(all_obs).reshape(self.n_agents, -1)

        for i, obs in enumerate(observations):
            if self.enable_communication:
                # Concatenate observation with message (detach to avoid gradient issues)
                obs_with_msg = torch.cat([torch.FloatTensor(obs), messages[i].detach()])
                action = self.agents[i].act(obs_with_msg.numpy(), epsilon)
            else:
                action = self.agents[i].act(obs, epsilon)
            actions.append(action)

        return actions

    def update(self, experiences):
        """Update all agents"""
        for i, exp in enumerate(experiences):
            if len(self.agents[i].replay_buffer) > 32:
                self.agents[i].update(batch_size=32)
