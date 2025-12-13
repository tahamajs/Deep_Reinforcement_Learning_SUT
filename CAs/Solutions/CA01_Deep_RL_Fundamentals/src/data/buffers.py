import numpy as np
import torch
import random
from collections import deque, namedtuple
from typing import Tuple, List, Optional, Any


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
        states = torch.from_numpy(states).float().to("cpu")
        actions = torch.from_numpy(actions).long().to("cpu")
        rewards = torch.from_numpy(rewards).float().to("cpu")
        next_states = torch.from_numpy(next_states).float().to("cpu")
        dones = torch.from_numpy(dones).float().to("cpu")
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0

    def add(
        self, state, action, reward, next_state, done, priority: Optional[float] = None
    ) -> None:
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        if priority is None:
            max_prio = self.priorities.max() if self.pos > 0 else 1.0
            prio = max_prio
        else:
            prio = float(priority)
        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            raise ValueError("The buffer is empty")
        prios = self.priorities[: len(self.buffer)]
        probs = prios**self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        is_weights = torch.from_numpy(weights).float().unsqueeze(1).to("cpu")
        states = (
            torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to("cpu")
        )
        actions = (
            torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to("cpu")
        )
        rewards = (
            torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to("cpu")
        )
        next_states = (
            torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to("cpu")
        )
        dones = (
            torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8))
            .float()
            .to("cpu")
        )
        return (states, actions, rewards, next_states, dones), indices, is_weights

    def update_priorities(self, indices, priorities) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio) + 1e-6

    def __len__(self) -> int:
        return len(self.buffer)


