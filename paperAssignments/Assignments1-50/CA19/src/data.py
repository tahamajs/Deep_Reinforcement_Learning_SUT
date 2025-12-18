from typing import Deque, Dict, Any, NamedTuple, Tuple
from collections import deque
import random
import torch


class Transition(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    reward: float
    next_obs: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ):
        self.buffer.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = random.sample(list(self.buffer), k=batch_size)
        obs = torch.stack([t.obs for t in batch])
        actions = torch.stack([t.action for t in batch]).long().squeeze(-1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_obs = torch.stack([t.next_obs for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }

    def __len__(self) -> int:
        return len(self.buffer)






