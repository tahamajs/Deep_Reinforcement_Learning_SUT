import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import random

class FederatedReplayBuffer:
    """A replay buffer adapted for federated settings.
    Manages multiple client buffers and can sample from them.
    """
    def __init__(self, capacity: int, n_clients: int, seed: int):
        self.capacity = capacity
        self.n_clients = n_clients
        self.buffers: List[List[Tuple[Any, ...]]] = [[] for _ in range(n_clients)]
        self.rng = random.Random(seed)

    def add(self, client_id: int, experience: Tuple[Any, ...]):
        if len(self.buffers[client_id]) < self.capacity:
            self.buffers[client_id].append(experience)
        else:
            self.buffers[client_id][self.rng.randint(0, self.capacity - 1)] = experience

    def sample(self, client_id: int, batch_size: int) -> List[Tuple[Any, ...]]:
        buffer = self.buffers[client_id]
        if len(buffer) < batch_size:
            return []
        return self.rng.sample(buffer, batch_size)

    def sample_global(self, batch_size: int) -> List[Tuple[Any, ...]]:
        all_experiences = [exp for buffer in self.buffers for exp in buffer]
        if len(all_experiences) < batch_size:
            return []
        return self.rng.sample(all_experiences, batch_size)

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)

class FederatedSafeCausalMountainCar(gym.Env):
    """Custom Gymnasium environment wrapper that simulates a federated, safe, and causal Mountain Car environment.
    This will manage multiple client environments and expose appropriate APIs for federated training.
    """
    def __init__(self, config: Any, client_id: int = 0):
        super().__init__()
        self.config = config
        self.client_id = client_id
        self.env = gym.make("MountainCarContinuous-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        
        # Introduce client-specific heterogeneity (e.g., slightly different gravity)
        self.gravity = self.env.unwrapped.gravity * (1.0 + self.config.client_data_heterogeneity * (client_id - self.config.n_clients / 2) / self.config.n_clients)
        
        # Define a safety cost: penalty for high velocity
        self.safety_velocity_threshold = 0.05 # Example threshold
        self.danger_zone_position_range = (-0.3, -0.1) # Example danger zone

        self.seed(config.seed + client_id) # Seed for reproducibility per client

    def seed(self, seed: int):
        self.env.reset(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _get_cost(self, observation: np.ndarray, action: np.ndarray) -> float:
        position, velocity = observation
        cost = 0.0
        # Cost for high velocity
        if abs(velocity) > self.safety_velocity_threshold:
            cost += self.config.cost_scale * (abs(velocity) - self.safety_velocity_threshold)
        # Cost for being in a danger zone
        if self.danger_zone_position_range[0] < position < self.danger_zone_position_range[1]:
            cost += self.config.cost_scale * 0.5 # A fixed penalty for being in the danger zone
        return cost

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply client-specific gravity (hacky way to modify env dynamics for demonstration)
        # In a real gym env, you would modify the underlying physics or use wrappers.
        self.env.unwrapped.gravity = self.gravity
        
        cost = self._get_cost(observation, action)
        
        # Add causal intervention info (placeholder)
        info["causal_intervention_possible"] = (observation[0] > 0.4) # Example causal trigger
        info["cost"] = cost # Add cost to info for logging

        return observation, reward, cost, terminated, truncated, info

    def reset(self, *, seed: Any = None, options: Any = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

class ClientDataset:
    """Represents the local dataset on each client."""
    def __init__(self, experiences: List[Tuple[Any, ...]]):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return self.experiences[idx]

    @staticmethod
    def collate_fn(batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, ...]:
        # This collate_fn assumes experiences are (obs, action, reward, cost, next_obs, terminated, truncated)
        obs, actions, rewards, costs, next_obs, terminateds, truncateds = zip(*batch)
        
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        costs = torch.tensor(np.array(costs), dtype=torch.float32).unsqueeze(1)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)
        terminateds = torch.tensor(np.array(terminateds), dtype=torch.float32).unsqueeze(1)
        truncateds = torch.tensor(np.array(truncateds), dtype=torch.float32).unsqueeze(1)
        
        return obs, actions, rewards, costs, next_obs, terminateds, truncateds







