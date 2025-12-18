from typing import Dict, Any
import numpy as np
import torch


class ReplayBuffer:
    """
    Minimal replay buffer that can be filled from numpy arrays (e.g., D4RL dataset)
    or appended to during training. Designed for offline RL usage.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add_batch(self, batch: Dict[str, Any]) -> None:
        n = batch["obs"].shape[0]
        idx = np.arange(self.ptr, self.ptr + n) % self.capacity
        self.obs[idx] = batch["obs"]
        self.next_obs[idx] = batch["next_obs"]
        self.acts[idx] = batch["actions"]
        self.rews[idx] = batch["rewards"].reshape(-1, 1)
        self.dones[idx] = batch.get("dones", np.zeros((n, 1))).reshape(-1, 1)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]).to(self.device),
            "next_obs": torch.as_tensor(self.next_obs[idx]).to(self.device),
            "actions": torch.as_tensor(self.acts[idx]).to(self.device),
            "rewards": torch.as_tensor(self.rews[idx]).to(self.device),
            "dones": torch.as_tensor(self.dones[idx]).to(self.device),
        }

    @classmethod
    def from_d4rl(cls, dataset: Dict[str, np.ndarray], device: str = "cpu"):
        """
        Build a ReplayBuffer from a D4RL-style dataset dict.
        Expected keys: 'observations','actions','rewards','next_observations','terminals'
        """
        obs = dataset["observations"].astype(np.float32)
        next_obs = dataset["next_observations"].astype(np.float32)
        acts = dataset["actions"].astype(np.float32)
        rews = dataset["rewards"].astype(np.float32).reshape(-1, 1)
        dones = (
            dataset.get("terminals", np.zeros((obs.shape[0], 1)))
            .astype(np.float32)
            .reshape(-1, 1)
        )
        buf = cls(
            capacity=obs.shape[0],
            obs_dim=obs.shape[1],
            act_dim=acts.shape[1],
            device=device,
        )
        buf.obs[: obs.shape[0]] = obs
        buf.next_obs[: obs.shape[0]] = next_obs
        buf.acts[: acts.shape[0]] = acts
        buf.rews[: rews.shape[0]] = rews
        buf.dones[: dones.shape[0]] = dones
        buf.size = obs.shape[0]
        buf.ptr = buf.size % buf.capacity
        return buf


