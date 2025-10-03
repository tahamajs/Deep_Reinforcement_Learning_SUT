import random
from collections import deque, namedtuple
from typing import Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..models.ca1_models import DQN, DuelingDQN
from ..utils.ca1_utils import device


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"]
        )

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences = random.sample(self.buffer, k=batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(
            np.uint8
        )

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        update_every: int = 4,
        tau: float = 1e-3,
        use_double_dqn: bool = False,
        use_dueling: bool = False,
    ) -> None:

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.use_double_dqn = use_double_dqn

        if use_dueling:
            self.q_network = DuelingDQN(state_size, action_size).to(device)
            self.target_network = DuelingDQN(state_size, action_size).to(device)
        else:
            self.q_network = DQN(state_size, action_size).to(device)
            self.target_network = DQN(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
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
        if eps is None:
            eps = self.epsilon

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
    ) -> None:
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

    def soft_update(
        self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float
    ) -> None:
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def hard_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class PolicyNetwork(torch.nn.Module):
    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 64
    ) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class REINFORCEAgent:
    def __init__(
        self, state_size: int, action_size: int, lr: float = 1e-3, gamma: float = 0.99
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []

    def act(self, state: np.ndarray) -> int:
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

    def learn(self) -> float:
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
        return float(loss.item())


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 64) -> None:
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.actor = PolicyNetwork(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = ValueNetwork(state_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
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
    ) -> Tuple[float, float]:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state_t = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        reward_t = torch.tensor([reward]).float().to(device)
        done_t = torch.tensor([done]).float().to(device)

        current_value = self.critic(state_t)
        next_value = (
            self.critic(next_state_t)
            if not done
            else torch.zeros_like(current_value).to(device)
        )

        td_target = reward_t + self.gamma * next_value * (1 - done_t)
        td_error = td_target - current_value

        critic_loss = td_error.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = (-log_prob * td_error.detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return float(actor_loss.item()), float(critic_loss.item())


def train_dqn_agent(agent: DQNAgent, env, n_episodes: int = 1000, max_t: int = 1000):
    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        state = np.array(state, dtype=np.float32)
        score = 0.0

        for t in range(max_t):
            action = agent.act(state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.3f}"
            )
        if (
            len(scores_window) == scores_window.maxlen
            and np.mean(scores_window) >= 195.0
        ):
            print(
                f"Environment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
            )
            break
    return scores


def train_reinforce_agent(
    agent: REINFORCEAgent, env, n_episodes: int = 1000, max_t: int = 1000
):
    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        state = np.array(state, dtype=np.float32)
        agent.reset_episode()
        score = 0.0

        for t in range(max_t):
            action = agent.act(state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)
            agent.step(state, action, reward)
            state = next_state
            score += reward
            if done:
                break

        loss = agent.learn()
        scores_window.append(score)
        scores.append(score)
        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tLoss: {loss:.3f}"
            )
        if (
            len(scores_window) == scores_window.maxlen
            and np.mean(scores_window) >= 195.0
        ):
            print(
                f"Environment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
            )
            break
    return scores


def train_actor_critic_agent(
    agent: ActorCriticAgent, env, n_episodes: int = 1000, max_t: int = 1000
):
    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        state = np.array(state, dtype=np.float32)
        score = 0.0

        for t in range(max_t):
            action, log_prob = agent.act(state)
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            actor_loss, critic_loss = agent.learn(
                state, action, reward, next_state, done, log_prob
            )
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tActorLoss: {actor_loss:.3f}\tCriticLoss: {critic_loss:.3f}"
            )
        if (
            len(scores_window) == scores_window.maxlen
            and np.mean(scores_window) >= 195.0
        ):
            print(
                f"Environment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
            )
            break
    return scores
