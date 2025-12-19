import numpy as np
from typing import List, Tuple

class PrioritizedReplayBuffer:
    """A simple proportional Prioritized Experience Replay implementation.

    This is a compact, numpy-based implementation intended for educational
    purposes and small experiments. For large-scale use, prefer a C-optimized
    implementation or libraries with SumTree support.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.buffer = []
        # store priority values; initialized with 1.0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.priorities.max() if len(self.buffer) > 0 else 1.0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List, np.ndarray, np.ndarray]:
        if batch_size > len(self.buffer):
            raise ValueError(f"Requested batch_size={batch_size} but only {len(self.buffer)} elements in buffer")
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            probs[indices]
        )

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        # Clip errors and set priorities
        eps = 1e-6
        self.priorities[indices] = np.abs(errors) + eps

    def __len__(self):
        return len(self.buffer)