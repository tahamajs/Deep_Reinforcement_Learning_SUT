import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.cost = np.zeros((size, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done, cost):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.cost[self.ptr] = cost
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.act[idx], device=self.device),
            torch.as_tensor(self.rew[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
            torch.as_tensor(self.cost[idx], device=self.device),
        )
