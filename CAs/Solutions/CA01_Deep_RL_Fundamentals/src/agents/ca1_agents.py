import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod

from ..models.ca1_models import DQN, DuelingDQN, PolicyNetwork, ValueNetwork
from ..data.buffers import ReplayBuffer, PrioritizedReplayBuffer
from ..config import DQNConfig, REINFORCEConfig, ActorCriticConfig

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state: np.ndarray, eps: Optional[float] = None) -> int:
        pass

    @abstractmethod
    def learn(self, *args, **kwargs) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class DQNAgent(BaseAgent):
    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.batch_size = config.batch_size
        self.update_every = config.update_every
        self.tau = config.tau
        self.use_double_dqn = config.use_double_dqn
        if config.use_dueling:
            self.q_network = DuelingDQN(config.state_size, config.action_size, config.hidden_size).to(device)
            self.target_network = DuelingDQN(config.state_size, config.action_size, config.hidden_size).to(device)
        else:
            self.q_network = DQN(config.state_size, config.action_size, config.hidden_size).to(device)
            self.target_network = DQN(config.state_size, config.action_size, config.hidden_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.buffer_size)
        self.t_step = 0
        self.hard_update(self.target_network, self.q_network)

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def act(self, state: np.ndarray, eps: Optional[float] = None) -> int:
        eps = eps if eps is not None else self.epsilon
        if random.random() > eps:
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state_t)
            self.q_network.train()
            return int(action_values.argmax(dim=1).item())
        else:
            return int(random.choice(np.arange(self.action_size)))

    def learn(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = experiences
        if self.use_double_dqn:
            next_actions = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
            Q_targets_next = (
                self.target_network(next_states).detach().gather(1, next_actions)
            )
        else:
            Q_targets_next = (
                self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            )
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.q_network, self.target_network, self.tau)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"loss": loss.item(), "epsilon": self.epsilon}

    def soft_update(
        self, local_model: nn.Module, target_model: nn.Module, tau: float
    ) -> None:
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def hard_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "epsilon": self.epsilon,
            },
            path,
        )
        logger.info(f"Agent saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config = checkpoint["config"]
        self.epsilon = checkpoint["epsilon"]
        logger.info(f"Agent loaded from {path}")


class ImprovedDQNAgent(DQNAgent):
    def __init__(self, config: DQNConfig) -> None:
        super().__init__(config)
        self.memory = PrioritizedReplayBuffer(config.buffer_size)

    def learn(self) -> Dict[str, float]:
        experiences, indices, weights = (
            self.memory.sample(self.config.batch_size)
        )
        states, actions, rewards, next_states, dones = experiences

        if self.config.use_double_dqn:
            next_actions = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
            Q_targets_next = (
                self.target_network(next_states).detach().gather(1, next_actions)
            )
        else:
            Q_targets_next = (
                self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            )
        Q_targets = rewards + (self.config.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        td_errors = (Q_targets - Q_expected).detach().squeeze().abs().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update(self.q_network, self.target_network, self.config.tau)
        self.epsilon = max(
            self.config.epsilon_min, self.epsilon * self.config.epsilon_decay
        )
        return {"loss": loss.item(), "epsilon": self.epsilon, "td_error_mean": td_errors.mean()}


class REINFORCEAgent(BaseAgent):
    def __init__(self, config: REINFORCEConfig) -> None:
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.gamma = config.gamma
        self.policy = PolicyNetwork(config.state_size, config.action_size, config.hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []

    def act(self, state: np.ndarray, eps: Optional[float] = None) -> int:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state_t)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return int(action.item())

    def step(self, state: np.ndarray, action: int, reward: float) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self) -> Dict[str, float]:
        returns: List[float] = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns).float().to(device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        policy_loss = []
        for log_prob, Gt in zip(self.log_probs, returns_t):
            policy_loss.append(-log_prob * Gt)
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.reset_episode()
        return {"loss": float(loss.item())}
    
    def save(self, path: str) -> None:
        torch.save({
            "policy_network": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"REINFORCE agent saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config = checkpoint["config"]
        logger.info(f"REINFORCE agent loaded from {path}")


class ActorCriticAgent(BaseAgent):
    def __init__(self, config: ActorCriticConfig) -> None:
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.gamma = config.gamma
        self.actor = PolicyNetwork(config.state_size, config.action_size, config.hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic = ValueNetwork(config.state_size, config.hidden_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

    def act(self, state: np.ndarray, eps: Optional[float] = None) -> Tuple[int, torch.Tensor]:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.actor(state_t)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return int(action.item()), m.log_prob(action)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: torch.Tensor,
    ) -> Dict[str, float]:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state_t = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        reward_t = torch.tensor([reward]).float().to(device)
        done_t = torch.tensor([done]).float().to(device)

        # Critic update
        current_value = self.critic(state_t)
        next_value = (
            self.critic(next_state_t)
            if not done_t.item()
            else torch.zeros_like(current_value).to(device)
        )
        td_target = reward_t + self.gamma * next_value
        td_error = td_target - current_value
        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = (-log_prob * td_error.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}
    
    def save(self, path: str) -> None:
        torch.save({
            "actor_network": self.actor.state_dict(),
            "critic_network": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Actor-Critic agent saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_network"])
        self.critic.load_state_dict(checkpoint["critic_network"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.config = checkpoint["config"]
        logger.info(f"Actor-Critic agent loaded from {path}")

