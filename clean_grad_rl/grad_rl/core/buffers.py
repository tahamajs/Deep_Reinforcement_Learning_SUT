from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray | int | float
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, action_shape=(), prioritized: bool = False, alpha: float = 0.6):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        if action_shape == ():
            self.actions = np.zeros((capacity,), dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

        self.prioritized = prioritized
        self.alpha = alpha
        self.eps = 1e-6
        self.priorities = np.ones((capacity,), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done
        if self.prioritized:
            max_prio = self.priorities.max() if self.idx > 0 or self.full else 1.0
            self.priorities[self.idx] = max_prio

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int, beta: float = 0.4):
        size = len(self)
        if self.prioritized:
            probs = self.priorities[:size] ** self.alpha
            probs = probs / probs.sum()
            indices = np.random.choice(size, batch_size, p=probs)
            weights = (size * probs[indices]) ** (-beta)
            weights = weights / weights.max()
        else:
            indices = np.random.randint(0, size, size=batch_size)
            weights = np.ones((batch_size,), dtype=np.float32)

        batch = {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights.astype(np.float32),
        }
        return batch

    def update_priorities(self, indices, td_errors):
        if not self.prioritized:
            return
        self.priorities[indices] = np.abs(td_errors) + self.eps


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque(maxlen=n_step)

    def push(self, tr: Transition):
        self.buffer.append(tr)

    def ready(self) -> bool:
        return len(self.buffer) == self.n_step

    def pop_nstep(self):
        reward = 0.0
        next_obs = self.buffer[-1].next_obs
        done = self.buffer[-1].done
        obs = self.buffer[0].obs
        action = self.buffer[0].action
        for i, tr in enumerate(self.buffer):
            reward += (self.gamma ** i) * tr.reward
            if tr.done:
                next_obs = tr.next_obs
                done = tr.done
                break
        first = self.buffer.popleft()
        return obs, action, reward, next_obs, done
