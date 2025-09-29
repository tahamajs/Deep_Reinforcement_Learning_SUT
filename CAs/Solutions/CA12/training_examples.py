"""
Multi-Agent Reinforcement Learning and Advanced Policy Methods - Training Examples
==================================================================================

This module provides comprehensive implementations and training examples for
Multi-Agent Reinforcement Learning and Advanced Policy Methods (CA12).

Key Components:
- Multi-Agent Actor-Critic (MAAC) methods
- Value Decomposition Networks (VDN)
- Counterfactual Multi-Agent Policy Gradients (COMA)
- Proximal Policy Optimization (PPO) variants
- Asynchronous Advantage Actor-Critic (A3C)
- Emergent communication protocols

Author: DRL Course Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import gymnasium as gym
from collections import defaultdict, deque
import random
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============================================================================
# MULTI-AGENT ACTOR-CRITIC (MAAC)
# =============================================================================

class MultiAgentActor(nn.Module):
    """Actor network for multi-agent settings"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Sample action from policy"""
        logits = self.forward(obs)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)

        return action.item(), log_prob

class MultiAgentCritic(nn.Module):
    """Centralized critic for multi-agent settings"""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Centralized critic takes all observations and actions
        total_input_dim = obs_dim * num_agents + action_dim * num_agents

        self.network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass - obs and actions are concatenated"""
        x = torch.cat([obs.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=-1)
        return self.network(x)

class MAACAgent:
    """Multi-Agent Actor-Critic agent"""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 num_agents: int,
                 hidden_dim: int = 128,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau

        # Actor networks (one per agent)
        self.actors = [MultiAgentActor(obs_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.actor_targets = [MultiAgentActor(obs_dim, action_dim, hidden_dim) for _ in range(num_agents)]

        # Copy parameters to targets
        for actor, target in zip(self.actors, self.actor_targets):
            target.load_state_dict(actor.state_dict())

        # Centralized critic
        self.critic = MultiAgentCritic(obs_dim, action_dim, num_agents, hidden_dim)
        self.critic_target = MultiAgentCritic(obs_dim, action_dim, num_agents, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = deque(maxlen=100000)

    def select_actions(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[List[int], torch.Tensor]:
        """Select actions for all agents"""
        actions = []
        log_probs = []

        for i, agent_obs in enumerate(obs):
            action, log_prob = self.actors[i].get_action(agent_obs.unsqueeze(0), deterministic)
            actions.append(action)
            log_probs.append(log_prob)

        return actions, torch.stack(log_probs)

    def store_transition(self, obs: torch.Tensor, actions: List[int], reward: float,
                        next_obs: torch.Tensor, done: bool):
        """Store transition in buffer"""
        self.buffer.append({
            'obs': obs,
            'actions': actions,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update MAAC agent"""
        if len(self.buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(list(self.buffer), batch_size)
        obs_batch = torch.stack([t['obs'] for t in batch])
        actions_batch = torch.tensor([t['actions'] for t in batch])
        rewards_batch = torch.tensor([t['reward'] for t in batch], dtype=torch.float32)
        next_obs_batch = torch.stack([t['next_obs'] for t in batch])
        dones_batch = torch.tensor([t['done'] for t in batch], dtype=torch.float32)

        # Compute targets
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = []
            for i in range(self.num_agents):
                next_action_logits = self.actor_targets[i](next_obs_batch[:, i])
                next_action = torch.argmax(next_action_logits, dim=-1)
                next_actions.append(next_action)
            next_actions = torch.stack(next_actions, dim=1)

            # Compute target Q-value
            target_q = self.critic_target(next_obs_batch, next_actions)
            targets = rewards_batch.unsqueeze(-1) + self.gamma * (1 - dones_batch.unsqueeze(-1)) * target_q

        # Update critic
        self.critic_optimizer.zero_grad()
        current_q = self.critic(obs_batch, actions_batch)
        critic_loss = F.mse_loss(current_q, targets)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actors
        actor_losses = []
        for i in range(self.num_agents):
            self.actor_optimizers[i].zero_grad()

            # Get current actions from actor i
            actor_logits = self.actors[i](obs_batch[:, i])
            actor_actions = torch.argmax(actor_logits, dim=-1)

            # Create action tensor with actor i's action
            actions_with_actor = actions_batch.clone()
            actions_with_actor[:, i] = actor_actions

            # Compute actor loss
            actor_q = self.critic(obs_batch, actions_with_actor)
            actor_loss = -actor_q.mean()

            actor_loss.backward()
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())

        # Soft update targets
        for actor, target in zip(self.actors, self.actor_targets):
            for param, target_param in zip(actor.parameters(), target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss_avg': np.mean(actor_losses)
        }

# =============================================================================
# VALUE DECOMPOSITION NETWORKS (VDN)
# =============================================================================

class VDNNetwork(nn.Module):
    """Value Decomposition Network"""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        # Individual agent value networks
        self.agent_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_agents)
        ])

        # Mixing network for value decomposition
        self.mixing_network = nn.Sequential(
            nn.Linear(num_agents * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute Q-values for each agent"""
        agent_qs = []
        for i in range(self.num_agents):
            q_values = self.agent_networks[i](obs[:, i])
            agent_qs.append(q_values)

        # Mix Q-values
        agent_qs = torch.stack(agent_qs, dim=1)  # [batch, agents, actions]
        agent_qs_flat = agent_qs.view(agent_qs.size(0), -1)  # [batch, agents * actions]
        mixed_q = self.mixing_network(agent_qs_flat)

        return mixed_q, agent_qs

class VDNAgent:
    """Value Decomposition Network agent"""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 num_agents: int,
                 hidden_dim: int = 128,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.network = VDNNetwork(obs_dim, action_dim, num_agents, hidden_dim)
        self.target_network = VDNNetwork(obs_dim, action_dim, num_agents, hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = deque(maxlen=100000)

    def select_actions(self, obs: torch.Tensor) -> List[int]:
        """Select actions using epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return [np.random.randint(self.action_dim) for _ in range(self.num_agents)]
        else:
            with torch.no_grad():
                _, agent_qs = self.network(obs.unsqueeze(0))
                actions = torch.argmax(agent_qs.squeeze(0), dim=-1).tolist()
            return actions

    def store_transition(self, obs: torch.Tensor, actions: List[int], reward: float,
                        next_obs: torch.Tensor, done: bool):
        """Store transition"""
        self.buffer.append({
            'obs': obs,
            'actions': actions,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update VDN agent"""
        if len(self.buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(list(self.buffer), batch_size)
        obs_batch = torch.stack([t['obs'] for t in batch])
        actions_batch = torch.tensor([t['actions'] for t in batch])
        rewards_batch = torch.tensor([t['reward'] for t in batch], dtype=torch.float32)
        next_obs_batch = torch.stack([t['next_obs'] for t in batch])
        dones_batch = torch.tensor([t['done'] for t in batch], dtype=torch.float32)

        # Compute targets
        with torch.no_grad():
            next_mixed_q, _ = self.target_network(next_obs_batch)
            targets = rewards_batch.unsqueeze(-1) + self.gamma * (1 - dones_batch.unsqueeze(-1)) * next_mixed_q

        # Compute current Q-values
        current_mixed_q, _ = self.network(obs_batch)

        # Update
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_mixed_q, targets)
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Soft update target
        tau = 0.01
        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {'vdn_loss': loss.item(), 'epsilon': self.epsilon}

# =============================================================================
# COUNTERFACTUAL MULTI-AGENT POLICY GRADIENTS (COMA)
# =============================================================================

class COMACritic(nn.Module):
    """Centralized critic for COMA"""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        total_input_dim = obs_dim * num_agents + action_dim * num_agents

        self.network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)  # One advantage per agent
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = torch.cat([obs.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=-1)
        return self.network(x)

class COMAAgent:
    """Counterfactual Multi-Agent Policy Gradients agent"""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 num_agents: int,
                 hidden_dim: int = 128,
                 gamma: float = 0.99,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma

        # Actor networks
        self.actors = [MultiAgentActor(obs_dim, action_dim, hidden_dim) for _ in range(num_agents)]

        # Centralized critic
        self.critic = COMACritic(obs_dim, action_dim, num_agents, hidden_dim)

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = deque(maxlen=100000)

    def select_actions(self, obs: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """Select actions for all agents"""
        actions = []
        log_probs = []

        for i, agent_obs in enumerate(obs):
            action, log_prob = self.actors[i].get_action(agent_obs.unsqueeze(0))
            actions.append(action)
            log_probs.append(log_prob)

        return actions, torch.stack(log_probs)

    def store_transition(self, obs: torch.Tensor, actions: List[int], reward: float,
                        next_obs: torch.Tensor, done: bool):
        """Store transition"""
        self.buffer.append({
            'obs': obs,
            'actions': actions,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update COMA agent"""
        if len(self.buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(list(self.buffer), batch_size)
        obs_batch = torch.stack([t['obs'] for t in batch])
        actions_batch = torch.tensor([t['actions'] for t in batch])
        rewards_batch = torch.tensor([t['reward'] for t in batch], dtype=torch.float32)
        next_obs_batch = torch.stack([t['next_obs'] for t in batch])
        dones_batch = torch.tensor([t['done'] for t in batch], dtype=torch.float32)

        # Update critic
        self.critic_optimizer.zero_grad()
        advantages = self.critic(obs_batch, actions_batch)
        critic_loss = F.mse_loss(advantages, rewards_batch.unsqueeze(-1).expand(-1, self.num_agents))
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actors using counterfactual advantages
        actor_losses = []
        for i in range(self.num_agents):
            self.actor_optimizers[i].zero_grad()

            # Compute counterfactual advantage for agent i
            # Q(s, a_{-i}, a_i) - Q(s, a_{-i}, a'_i) where a'_i is from baseline
            advantages_i = advantages[:, i]

            # Get log probs for current actions
            current_log_probs = []
            for j, agent_obs in enumerate(obs_batch[:, i]):
                logits = self.actors[i](agent_obs.unsqueeze(0))
                log_prob = F.log_softmax(logits, dim=-1)[0, actions_batch[j, i]]
                current_log_probs.append(log_prob)

            current_log_probs = torch.stack(current_log_probs)

            # COMA loss
            actor_loss = -(current_log_probs * advantages_i.detach()).mean()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss_avg': np.mean(actor_losses)
        }

# =============================================================================
# ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
# =============================================================================

class A3CNetwork(nn.Module):
    """Shared network for A3C"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared feature layers
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.features(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

class A3CAgent:
    """Asynchronous Advantage Actor-Critic agent"""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 gamma: float = 0.99,
                 tau: float = 1.0,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        self.network = A3CNetwork(obs_dim, action_dim, hidden_dim)

    def select_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action"""
        logits, value = self.network(obs.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action])
        return action, log_prob, value.squeeze()

    def compute_returns(self, rewards: List[float], values: List[float], done: bool) -> torch.Tensor:
        """Compute n-step returns"""
        returns = []
        R = values[-1] if not done else 0

        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32)

    def update(self, obs: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor,
               returns: torch.Tensor, values: torch.Tensor) -> Dict[str, float]:
        """Update A3C agent"""

        # Get current outputs
        logits, current_values = self.network(obs)
        current_values = current_values.squeeze()

        # Policy loss
        advantages = returns - current_values.detach()
        new_log_probs = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(new_log_probs - log_probs)
        policy_loss = -(torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages)).mean()

        # Value loss
        value_loss = F.mse_loss(current_values, returns)

        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Total loss
        total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }

# =============================================================================
# EMERGENT COMMUNICATION
# =============================================================================

class CommunicationAgent(nn.Module):
    """Agent with communication capabilities"""

    def __init__(self, obs_dim: int, action_dim: int, message_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.message_dim = message_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Communication encoder (receives messages from other agents)
        self.comm_encoder = nn.Sequential(
            nn.Linear(message_dim * 2, hidden_dim),  # Messages from 2 other agents
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Message generator
        self.message_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat obs and comm features
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor, messages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        obs_features = self.obs_encoder(obs)
        comm_features = self.comm_encoder(messages.flatten(start_dim=1))

        combined_features = torch.cat([obs_features, comm_features], dim=-1)

        action_logits = self.policy(combined_features)
        message = self.message_generator(obs_features)

        return action_logits, message

class EmergentCommunicationSystem:
    """Multi-agent system with emergent communication"""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int = 3,
                 message_dim: int = 8, hidden_dim: int = 128):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.message_dim = message_dim

        self.agents = [CommunicationAgent(obs_dim, action_dim, message_dim, hidden_dim)
                      for _ in range(num_agents)]

        self.optimizers = [optim.Adam(agent.parameters(), lr=1e-3) for agent in self.agents]

    def select_actions(self, obs: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """Select actions with communication"""
        # Initialize messages (could be learned or random)
        messages = torch.randn(self.num_agents, self.num_agents - 1, self.message_dim)

        actions = []
        all_messages = []

        for i, agent in enumerate(self.agents):
            # Get messages from other agents (exclude self)
            agent_messages = torch.cat([messages[j] for j in range(self.num_agents) if j != i], dim=0)

            logits, message = agent(obs[i].unsqueeze(0), agent_messages.unsqueeze(0))
            action = torch.argmax(logits, dim=-1).item()

            actions.append(action)
            all_messages.append(message.squeeze(0))

        return actions, torch.stack(all_messages)

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_maac_agent(env_name: str = 'CartPole-v1',
                    num_agents: int = 2,
                    num_episodes: int = 500,
                    max_steps: int = 200,
                    seed: int = 42) -> Dict[str, Any]:
    """Train MAAC agent"""

    set_seed(seed)

    # For simplicity, use multiple independent environments
    envs = [gym.make(env_name) for _ in range(num_agents)]
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n

    agent = MAACAgent(obs_dim, action_dim, num_agents)

    episode_rewards = []
    critic_losses = []
    actor_losses = []

    print(f"Training MAAC Agent ({num_agents} agents) on {env_name}")
    print("=" * 50)

    for episode in tqdm(range(num_episodes)):
        # Reset environments
        obs = torch.stack([torch.tensor(env.reset()[0], dtype=torch.float32) for env in envs])
        episode_reward = 0

        for step in range(max_steps):
            actions, log_probs = agent.select_actions(obs)

            # Step environments
            next_obs = []
            rewards = []
            dones = []

            for i, env in enumerate(envs):
                next_o, r, terminated, truncated, _ = env.step(actions[i])
                next_obs.append(torch.tensor(next_o, dtype=torch.float32))
                rewards.append(r)
                dones.append(terminated or truncated)

            next_obs = torch.stack(next_obs)
            reward = sum(rewards) / len(rewards)  # Average reward
            done = any(dones)

            agent.store_transition(obs, actions, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        # Update agent
        if len(agent.buffer) > 100:
            loss_dict = agent.update()
            if loss_dict:
                critic_losses.append(loss_dict['critic_loss'])
                actor_losses.append(loss_dict['actor_loss_avg'])

    for env in envs:
        env.close()

    results = {
        'episode_rewards': episode_rewards,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'agent': agent,
        'config': {
            'env_name': env_name,
            'num_agents': num_agents,
            'num_episodes': num_episodes
        }
    }

    return results

def train_vdn_agent(env_name: str = 'CartPole-v1',
                   num_agents: int = 2,
                   num_episodes: int = 500,
                   max_steps: int = 200,
                   seed: int = 42) -> Dict[str, Any]:
    """Train VDN agent"""

    set_seed(seed)

    envs = [gym.make(env_name) for _ in range(num_agents)]
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n

    agent = VDNAgent(obs_dim, action_dim, num_agents)

    episode_rewards = []
    losses = []
    epsilons = []

    print(f"Training VDN Agent ({num_agents} agents) on {env_name}")
    print("=" * 45)

    for episode in tqdm(range(num_episodes)):
        obs = torch.stack([torch.tensor(env.reset()[0], dtype=torch.float32) for env in envs])
        episode_reward = 0

        for step in range(max_steps):
            actions = agent.select_actions(obs)

            next_obs = []
            rewards = []
            dones = []

            for i, env in enumerate(envs):
                next_o, r, terminated, truncated, _ = env.step(actions[i])
                next_obs.append(torch.tensor(next_o, dtype=torch.float32))
                rewards.append(r)
                dones.append(terminated or truncated)

            next_obs = torch.stack(next_obs)
            reward = sum(rewards) / len(rewards)
            done = any(dones)

            agent.store_transition(obs, actions, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        # Update agent
        if len(agent.buffer) > 100:
            loss_dict = agent.update()
            if loss_dict:
                losses.append(loss_dict['vdn_loss'])
                epsilons.append(loss_dict['epsilon'])

    for env in envs:
        env.close()

    results = {
        'episode_rewards': episode_rewards,
        'losses': losses,
        'epsilons': epsilons,
        'agent': agent,
        'config': {
            'env_name': env_name,
            'num_agents': num_agents,
            'num_episodes': num_episodes
        }
    }

    return results

def compare_multi_agent_methods(env_name: str = 'CartPole-v1',
                               num_agents: int = 2,
                               num_runs: int = 3,
                               num_episodes: int = 200) -> Dict[str, Any]:
    """Compare different multi-agent methods"""

    methods = ['MAAC', 'VDN', 'Independent Q-Learning']
    results = {}

    for method in methods:
        print(f"Testing {method}...")

        run_rewards = []

        for run in range(num_runs):
            set_seed(42 + run)

            if method == 'MAAC':
                result = train_maac_agent(env_name, num_agents, num_episodes, seed=42 + run)
            elif method == 'VDN':
                result = train_vdn_agent(env_name, num_agents, num_episodes, seed=42 + run)
            else:  # Independent Q-Learning
                # Simplified independent learning
                envs = [gym.make(env_name) for _ in range(num_agents)]
                rewards = []
                for episode in range(num_episodes):
                    episode_reward = 0
                    for env in envs:
                        obs, _ = env.reset()
                        for step in range(200):
                            action = env.action_space.sample()
                            obs, r, terminated, truncated, _ = env.step(action)
                            episode_reward += r / num_agents
                            if terminated or truncated:
                                break
                    rewards.append(episode_reward)
                for env in envs:
                    env.close()
                result = {'episode_rewards': rewards}

            run_rewards.append(result['episode_rewards'])

        # Average across runs
        avg_rewards = np.mean(run_rewards, axis=0)
        std_rewards = np.std(run_rewards, axis=0)

        results[method] = {
            'mean_rewards': avg_rewards,
            'std_rewards': std_rewards,
            'final_score': np.mean(avg_rewards[-50:])  # Average of last 50 episodes
        }

    return results

# =============================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================

def analyze_multi_agent_coordination(save_path: Optional[str] = None) -> plt.Figure:
    """Analyze multi-agent coordination patterns"""

    print("Analyzing multi-agent coordination patterns...")
    print("=" * 50)

    # Simulate coordination data
    np.random.seed(42)
    n_episodes = 1000
    n_agents = 3

    # Different coordination scenarios
    scenarios = ['Independent', 'Communication', 'Centralized', 'Decentralized']

    coordination_data = {}

    for scenario in scenarios:
        if scenario == 'Independent':
            # Low coordination, high variance
            base_perf = 50 + 30 * (1 - np.exp(-np.arange(n_episodes) / 300))
            noise = np.random.normal(0, 15, n_episodes)
        elif scenario == 'Communication':
            # Good coordination with communication
            base_perf = 65 + 25 * (1 - np.exp(-np.arange(n_episodes) / 250))
            noise = np.random.normal(0, 8, n_episodes)
        elif scenario == 'Centralized':
            # High coordination, low variance
            base_perf = 70 + 20 * (1 - np.exp(-np.arange(n_episodes) / 200))
            noise = np.random.normal(0, 5, n_episodes)
        else:  # Decentralized
            # Moderate coordination
            base_perf = 60 + 28 * (1 - np.exp(-np.arange(n_episodes) / 280))
            noise = np.random.normal(0, 10, n_episodes)

        coordination_data[scenario] = base_perf + noise

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Coordination performance comparison
    episodes = np.arange(n_episodes)
    for scenario, scores in coordination_data.items():
        axes[0,0].plot(episodes, scores, linewidth=2, label=scenario, alpha=0.8)

    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Team Performance')
    axes[0,0].set_title('Multi-Agent Coordination Performance')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Coordination efficiency analysis
    coordination_efficiency = {}
    for scenario in scenarios:
        scores = coordination_data[scenario]
        # Measure how well agents coordinate (lower variance = better coordination)
        efficiency = 1 / (1 + np.var(scores[-200:]))  # Normalize
        coordination_efficiency[scenario] = efficiency

    colors = ['red', 'blue', 'green', 'orange']
    bars = axes[0,1].bar(range(len(coordination_efficiency)), list(coordination_efficiency.values()),
                         alpha=0.7, edgecolor='black', color=colors)
    axes[0,1].set_xlabel('Coordination Method')
    axes[0,1].set_ylabel('Coordination Efficiency')
    axes[0,1].set_title('Coordination Efficiency by Method')
    axes[0,1].set_xticks(range(len(coordination_efficiency)))
    axes[0,1].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)

    # Communication benefits
    comm_levels = [0, 0.2, 0.5, 0.8, 1.0]
    performance_gains = [0, 5, 15, 25, 35]  # Percentage improvement
    coordination_improvement = [0, 8, 20, 30, 40]

    axes[1,0].plot(comm_levels, performance_gains, 'o-', label='Performance Gain', linewidth=2, markersize=6)
    axes[1,0].plot(comm_levels, coordination_improvement, 's-', label='Coordination Improvement', linewidth=2, markersize=6)
    axes[1,0].set_xlabel('Communication Level')
    axes[1,0].set_ylabel('Improvement (%)')
    axes[1,0].set_title('Benefits of Communication in Multi-Agent Systems')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Scalability analysis
    agent_counts = [2, 3, 4, 5, 6]
    scalability_scores = {}

    for scenario in scenarios:
        if scenario == 'Independent':
            scores = [0.9, 0.8, 0.7, 0.6, 0.5]  # Scales poorly
        elif scenario == 'Communication':
            scores = [0.85, 0.8, 0.75, 0.7, 0.65]  # Moderate scaling
        elif scenario == 'Centralized':
            scores = [0.8, 0.6, 0.4, 0.3, 0.2]  # Poor scaling
        else:  # Decentralized
            scores = [0.88, 0.85, 0.82, 0.78, 0.75]  # Good scaling

        scalability_scores[scenario] = scores

    for scenario, scores in scalability_scores.items():
        axes[1,1].plot(agent_counts, scores, 'o-', label=scenario, linewidth=2, markersize=6)

    axes[1,1].set_xlabel('Number of Agents')
    axes[1,1].set_ylabel('Relative Performance')
    axes[1,1].set_title('Scalability Analysis by Coordination Method')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("Multi-agent coordination analysis completed!")

    return fig

def comprehensive_multi_agent_analysis(save_path: Optional[str] = None) -> Dict[str, Any]:
    """Comprehensive analysis of multi-agent RL methods"""

    print("Comprehensive multi-agent RL methods analysis...")
    print("=" * 55)

    methods = ['MAAC', 'VDN', 'QMIX', 'COMA', 'MADDPG', 'Independent Learning']
    environments = ['Cooperative', 'Competitive', 'Mixed', 'Communication-Heavy']

    # Method characteristics (1-10 scale)
    characteristics = {
        'Scalability': {
            'MAAC': 6, 'VDN': 8, 'QMIX': 7, 'COMA': 6, 'MADDPG': 5, 'Independent Learning': 9
        },
        'Sample Efficiency': {
            'MAAC': 7, 'VDN': 8, 'QMIX': 8, 'COMA': 7, 'MADDPG': 6, 'Independent Learning': 4
        },
        'Credit Assignment': {
            'MAAC': 6, 'VDN': 9, 'QMIX': 9, 'COMA': 8, 'MADDPG': 7, 'Independent Learning': 3
        },
        'Communication': {
            'MAAC': 5, 'VDN': 4, 'QMIX': 4, 'COMA': 5, 'MADDPG': 6, 'Independent Learning': 2
        }
    }

    # Performance by environment type
    performance_by_env = {
        'Cooperative': {
            'MAAC': 7, 'VDN': 9, 'QMIX': 9, 'COMA': 8, 'MADDPG': 7, 'Independent Learning': 5
        },
        'Competitive': {
            'MAAC': 6, 'VDN': 6, 'QMIX': 7, 'COMA': 8, 'MADDPG': 8, 'Independent Learning': 7
        },
        'Mixed': {
            'MAAC': 7, 'VDN': 8, 'QMIX': 8, 'COMA': 7, 'MADDPG': 7, 'Independent Learning': 6
        },
        'Communication-Heavy': {
            'MAAC': 6, 'VDN': 5, 'QMIX': 6, 'COMA': 6, 'MADDPG': 8, 'Independent Learning': 4
        }
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Method characteristics radar
    categories = list(characteristics.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for method in methods[:5]:  # Show first 5 to avoid clutter
        scores = [characteristics[cat][method] for cat in categories]
        scores += scores[:1]
        axes[0,0].plot(angles, scores, 'o-', linewidth=2, label=method, markersize=6)

    axes[0,0].set_xticks(angles[:-1])
    axes[0,0].set_xticklabels(categories, fontsize=9)
    axes[0,0].set_ylim(0, 10)
    axes[0,0].set_title('Multi-Agent Method Characteristics')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)

    # Performance by environment type
    env_names = list(performance_by_env.keys())
    x = np.arange(len(env_names))
    width = 0.15
    multiplier = 0

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (method, color) in enumerate(zip(methods, colors)):
        scores = [performance_by_env[env][method] for env in env_names]
        offset = width * multiplier
        bars = axes[0,1].bar(x + offset, scores, width, label=method, color=color, alpha=0.8)
        multiplier += 1

    axes[0,1].set_xlabel('Environment Type')
    axes[0,1].set_ylabel('Performance Score')
    axes[0,1].set_title('Method Performance by Environment Type')
    axes[0,1].set_xticks(x + width * 2.5, env_names)
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)

    # Scalability vs performance trade-off
    scalability = [characteristics['Scalability'][m] for m in methods]
    performance = [np.mean([performance_by_env[env][m] for env in env_names]) for m in methods]

    axes[1,0].scatter(scalability, performance, s=200, alpha=0.7, c='blue')
    for i, method in enumerate(methods):
        axes[1,0].annotate(method, (scalability[i], performance[i]),
                          xytext=(5, 5), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    axes[1,0].set_xlabel('Scalability')
    axes[1,0].set_ylabel('Average Performance')
    axes[1,0].set_title('Scalability vs Performance Trade-off')
    axes[1,0].grid(True, alpha=0.3)

    # Method evolution timeline
    years = [2017, 2018, 2018, 2018, 2019, 1990]
    method_timeline = ['MAAC', 'VDN', 'QMIX', 'COMA', 'MADDPG', 'Independent Learning']
    innovation_scores = [7, 8, 8, 7, 8, 5]

    axes[1,1].scatter(years, innovation_scores, s=150, alpha=0.7, c='green')
    for i, (year, method) in enumerate(zip(years, method_timeline)):
        axes[1,1].annotate(method, (year, innovation_scores[i]),
                          xytext=(5, 5), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    axes[1,1].set_xlabel('Year Introduced')
    axes[1,1].set_ylabel('Innovation Impact')
    axes[1,1].set_title('Multi-Agent Methods Timeline')
    axes[1,1].grid(True, alpha=0.3)

    # Strengths and limitations
    aspects = ['Strengths', 'Limitations']
    method_analysis = {
        'MAAC': [7, 5],
        'VDN': [8, 4],
        'QMIX': [8, 4],
        'COMA': [7, 5],
        'MADDPG': [7, 6],
        'Independent Learning': [4, 8]
    }

    x = np.arange(len(methods))
    width = 0.35

    for i, aspect in enumerate(aspects):
        scores = [method_analysis[method][i] for method in methods]
        axes[2,0].bar(x + (i-0.5)*width, scores, width, label=aspect, alpha=0.8)

    axes[2,0].set_xlabel('Method')
    axes[2,0].set_ylabel('Score (1-10)')
    axes[2,0].set_title('Method Strengths and Limitations')
    axes[2,0].set_xticks(x)
    axes[2,0].set_xticklabels(methods, rotation=45, ha='right')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)

    # Future directions
    future_areas = ['Hierarchical MARL', 'Meta-Learning', 'Continual MARL', 'Human-AI Coordination']
    current_state = [6, 5, 4, 7]
    potential_impact = [9, 8, 8, 9]

    x = np.arange(len(future_areas))
    width = 0.35

    axes[2,1].bar(x - width/2, current_state, width, label='Current State', alpha=0.7)
    axes[2,1].bar(x + width/2, potential_impact, width, label='Potential Impact', alpha=0.7)
    axes[2,1].set_xlabel('Research Area')
    axes[2,1].set_ylabel('Score (1-10)')
    axes[2,1].set_title('Future Directions for Multi-Agent RL')
    axes[2,1].set_xticks(x)
    axes[2,1].set_xticklabels(future_areas, rotation=45, ha='right')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print comprehensive analysis
    print("\n" + "=" * 55)
    print("MULTI-AGENT RL METHODS COMPREHENSIVE ANALYSIS")
    print("=" * 55)

    for method in methods:
        avg_perf = np.mean([performance_by_env[env][method] for env in env_names])
        print(f"{method:18} | Average Performance: {avg_perf:6.1f}")

    print("
💡 Key Insights for Multi-Agent RL:"    print("• Value decomposition methods excel in cooperative settings")
    print("• Communication improves coordination but reduces scalability")
    print("• Centralized training with decentralized execution is effective")
    print("• Credit assignment remains a key challenge")

    print("
🎯 Recommendations:"    print("• Use VDN/QMIX for cooperative multi-agent tasks")
    print("• Choose MADDPG for competitive or mixed environments")
    print("• Consider communication for complex coordination tasks")
    print("• Start with independent learning for simple problems")

    return {
        'characteristics': characteristics,
        'performance_by_env': performance_by_env,
        'methods': methods
    }

# =============================================================================
# MAIN TRAINING EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("Multi-Agent Reinforcement Learning and Advanced Policy Methods")
    print("=" * 70)
    print("Available training examples:")
    print("1. train_maac_agent() - Train Multi-Agent Actor-Critic")
    print("2. train_vdn_agent() - Train Value Decomposition Network")
    print("3. compare_multi_agent_methods() - Compare MARL methods")
    print("4. analyze_multi_agent_coordination() - Coordination analysis")
    print("5. comprehensive_multi_agent_analysis() - Full method comparison")
    print("\nExample usage:")
    print("results = train_maac_agent(num_agents=3)")
    print("comparison = compare_multi_agent_methods()")