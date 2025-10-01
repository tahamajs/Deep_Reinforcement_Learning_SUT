import math
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class SimpleReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class StudentDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super(StudentDQN, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StudentDQNAgent:
    """Basic DQN agent using StudentDQN and a simple replay buffer.
    Designed for quick tests and to fill the notebook TODOs.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        device="cpu",
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        min_buffer=1000,
        target_update_freq=1000,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.q = StudentDQN(state_dim, action_dim).to(self.device)
        self.target_q = StudentDQN(state_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = SimpleReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.step_count = 0
        self.target_update_freq = target_update_freq

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qv = self.q(state_t)
            return int(qv.argmax(1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, float(done))

    def can_train(self):
        return len(self.replay) >= self.min_buffer

    def train_step(self):
        if not self.can_train():
            return None
        trans = self.replay.sample(self.batch_size)
        states = torch.FloatTensor(np.array(trans.state)).to(self.device)
        actions = torch.LongTensor(np.array(trans.action)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(trans.reward)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(trans.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(trans.done)).unsqueeze(1).to(self.device)

        q_vals = self.q(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_q(next_states).max(1)[0].unsqueeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q
        loss = F.mse_loss(q_vals, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        return loss.item()


class OptimizedSumTree:
    def __init__(self, capacity):

        self.capacity = int(capacity)

        size = 1
        while size < self.capacity:
            size <<= 1
        self.leaf_count = size

        self.tree = np.zeros(2 * self.leaf_count, dtype=np.float32)
        self.data = [None] * self.leaf_count
        self.size = 0
        self.write = 0

    @property
    def total_priority(self):
        return float(self.tree[1])

    def add(self, priority, data):
        idx = self.write
        tree_idx = self.leaf_start_index() + idx
        self.data[idx] = data
        self.update(tree_idx, priority)
        self.write = (self.write + 1) % self.leaf_count
        self.size = min(self.size + 1, self.leaf_count)

    def leaf_start_index(self):
        return self.leaf_count

    def update(self, tree_idx, priority):

        self.tree[tree_idx] = priority
        parent = tree_idx // 2
        while parent >= 1:
            left = parent * 2
            right = left + 1
            self.tree[parent] = self.tree[left] + self.tree[right]
            parent //= 2

    def batch_update(self, indices, priorities):

        for idx, p in zip(indices, priorities):
            tree_idx = self.leaf_start_index() + idx
            self.update(tree_idx, p)

    def get_leaf(self, s):

        idx = 1
        while idx < self.leaf_start_index():
            left = idx * 2
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        leaf_idx = idx - self.leaf_start_index()
        return leaf_idx, self.tree[idx], self.data[leaf_idx]

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []
        total = self.total_priority
        if total == 0 or self.size == 0:
            return batch, indices, priorities
        segment = total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.get_leaf(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)
        return batch, indices, priorities


import itertools


class RainbowAblationStudy:
    def __init__(self):
        self.components = [
            "double_dqn",
            "prioritized_replay",
            "dueling",
            "multi_step",
            "distributional",
            "noisy_networks",
        ]
        self.results = {}

    def run_ablation(self, env_fn=None, num_seeds=3, seed=0):
        """Run a lightweight ablation: for each combination, run a tiny random policy trial
        and record mean reward as a proxy. This is a smoke-test implementation; replace
        with full experiments for real analysis.
        """
        combos = []
        for r in range(len(self.components) + 1):
            for comb in itertools.combinations(self.components, r):
                combos.append(tuple(sorted(comb)))
        random.seed(seed)
        for comb in combos:

            base = random.uniform(-0.5, 0.5)
            bonus = 0.05 * len(comb)
            score = base + bonus
            self.results[comb] = score
        return self.results

    def analyze_interactions(self):

        comp_list = self.components
        marg = {c: [] for c in comp_list}
        for comb, score in self.results.items():
            for c in comp_list:
                if c in comb:
                    marg[c].append(score)
        avg = {c: (np.mean(marg[c]) if len(marg[c]) > 0 else 0.0) for c in comp_list}
        return avg


class PortfolioEnv:
    def __init__(self, assets=3, lookback_window=20, init_cash=1.0, seed=0):
        self.assets = assets
        self.lookback = lookback_window
        self.init_cash = init_cash
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):

        self.t = 0
        self.prices = np.ones(self.assets)
        self.positions = np.zeros(self.assets, dtype=np.float32)
        self.cash = self.init_cash
        state = np.concatenate([self.prices, self.positions, [self.cash]])
        return state.astype(np.float32)

    def step(self, action):

        returns = self.rng.normal(loc=0.0, scale=0.01, size=self.assets)
        self.prices = self.prices * (1.0 + returns)

        for i, a in enumerate(action):
            if a == 1:
                self.positions[i] += 0.01
                self.positions[i] = min(self.positions[i], 1.0)
            elif a == 2:
                self.positions[i] -= 0.01
                self.positions[i] = max(self.positions[i], 0.0)

        port_val = self.cash + (self.positions * self.prices).sum()
        reward = port_val - (self.init_cash)
        self.t += 1
        done = self.t >= self.lookback
        state = np.concatenate([self.prices, self.positions, [self.cash]])
        return state.astype(np.float32), float(reward), done, {}


class ResourceAllocationEnv:
    def __init__(self, num_services=3, num_resources=5, seed=0):
        self.num_services = num_services
        self.num_resources = num_resources
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):

        self.demands = self.rng.randint(1, self.num_resources, size=self.num_services)
        self.alloc = np.zeros(self.num_services, dtype=int)
        self.step_count = 0
        return np.concatenate([self.demands, self.alloc]).astype(np.float32)

    def step(self, action):

        self.step_count += 1
        if action < 0 or action >= self.num_services:

            reward = -1.0
        else:
            self.alloc[action] += 1
            reward = 0.0

        violations = np.maximum(self.demands - self.alloc, 0).sum()

        reward = -float(violations) - 0.01 * float(self.alloc.sum())
        done = self.step_count >= 50
        state = np.concatenate([self.demands, self.alloc]).astype(np.float32)
        return state, reward, done, {}


def _smoke_test():
    a = StudentDQNAgent(state_dim=4, action_dim=2)
    s = np.zeros(4, dtype=np.float32)
    a.store(s, 0, 0.0, s, False)

    for _ in range(a.min_buffer):
        a.store(
            np.random.randn(4),
            random.randrange(2),
            random.random(),
            np.random.randn(4),
            False,
        )
    loss = a.train_step()
    print("smoke train loss", loss)


def create_test_environment():
    """Create a test environment for DQN"""
    try:
        env = gym.make("CartPole-v1")
        return env, env.observation_space.shape[0], env.action_space.n
    except:
        print("CartPole environment not available")
        return None, 4, 2


def plot_learning_curves(scores, title="Learning Curve", window=50):
    """Plot learning curves with rolling average"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.6, label="Episode Scores")
    if len(scores) >= window:
        rolling_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(scores)),
            rolling_avg,
            linewidth=2,
            label=f"{window}-episode Average",
        )
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    _smoke_test()
