"""
DQN and Double DQN (DDQN) implementation suitable for HW2 notebooks.

Provides:
- QNetwork (MLP)
- ReplayMemory
- DQNAgent with trainable methods

This file is import-safe and uses PyTorch if available. No heavy training is executed on import.
"""
from collections import deque, namedtuple
from typing import Deque, Tuple, Optional, List
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise ImportError("PyTorch is required for dqn_ddqn module") from e


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: Optional[torch.device] = None,
                 lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 64, mem_capacity: int = 10000,
                 target_update: int = 1000, double: bool = False):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayMemory(mem_capacity)
        self.steps_done = 0
        self.target_update = target_update
        self.double = double

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy selection. State is a 1D numpy array."""
        if random.random() < epsilon:
            return random.randrange(self.policy_net.net[-1].out_features)
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).to(self.device).unsqueeze(0)
            qvals = self.policy_net(s)
            return int(torch.argmax(qvals, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            if self.double:
                # Double DQN: select best action with policy_net, evaluate with target_net
                next_actions = torch.argmax(self.policy_net(next_state_batch), dim=1, keepdim=True)
                next_q = self.target_net(next_state_batch).gather(1, next_actions)
            else:
                next_q = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            expected_q = reward_batch + (1.0 - done_batch) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())


def train_dqn(env, agent: DQNAgent, episodes: int = 200, epsilon_start: float = 1.0,
              epsilon_end: float = 0.01, epsilon_decay: float = 0.995, max_steps: int = 1000):
    """Train loop for DQN/DDQN. Returns rewards per episode and losses per update."""
    rewards = []
    losses = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=None)
        state = np.array(obs, dtype=np.float32)
        total_r = 0.0
        done = False
        eps = max(epsilon_end, epsilon_start * (epsilon_decay ** ep))
        steps = 0
        while not done and steps < max_steps:
            action = agent.select_action(state, eps)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_obs, dtype=np.float32)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_r += reward
            steps += 1
        rewards.append(total_r)
    return np.array(rewards), np.array(losses)


if __name__ == "__main__":
    # Minimal usage example (will error if env/state dims not matching)
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode=None)
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, double=False)
    print("Created DQNAgent for CartPole")

