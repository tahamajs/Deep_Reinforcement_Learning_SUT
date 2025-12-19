import random
import numpy as np
import torch
from collections import deque
from typing import Tuple

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    This sets the Python, NumPy and PyTorch RNG seeds and configures
    deterministic CuDNN settings for reproducible results where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Prefer deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ReplayBuffer:
    """Experience replay buffer for DQN.

    Stores transitions as tuples (state, action, reward, next_state, done).
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences.

        Raises a ValueError if the requested batch size is larger than
        the number of stored transitions.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Requested batch_size={batch_size} but only {len(self.buffer)} elements in buffer")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)