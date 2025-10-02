import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, deque
import copy
import random
from sklearn.cluster import KMeans
from scipy import stats
import hashlib
class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning"""

    def __init__(
        self, epsilon: float = 1.0, delta: float = 1e-5, clipping_threshold: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold

    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Clip gradients to bound sensitivity"""
        grad_norm = torch.norm(gradients)
        clip_factor = min(1.0, self.clipping_threshold / grad_norm.item())
        return gradients * clip_factor

    def add_gaussian_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for differential privacy"""
        noise_scale = 2 * self.clipping_threshold / self.epsilon
        noise = torch.normal(0, noise_scale, gradients.shape)
        return gradients + noise

    def privatize_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply differential privacy to gradients"""
        clipped_grads = self.clip_gradients(gradients)
        private_grads = self.add_gaussian_noise(clipped_grads)
        return private_grads
class GradientCompression:
    """Compression techniques for efficient communication"""

    def __init__(self, compression_ratio: float = 0.1, quantization_levels: int = 256):
        self.compression_ratio = compression_ratio
        self.quantization_levels = quantization_levels

    def sparsify_top_k(
        self, gradients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep only top-k gradients by magnitude"""
        flat_grads = gradients.flatten()
        k = int(len(flat_grads) * self.compression_ratio)

        top_k_values, top_k_indices = torch.topk(torch.abs(flat_grads), k)

        sparse_grads = torch.zeros_like(flat_grads)
        sparse_grads[top_k_indices] = flat_grads[top_k_indices]

        return sparse_grads.reshape(gradients.shape), top_k_indices

    def quantize(self, gradients: torch.Tensor) -> torch.Tensor:
        """Quantize gradients to reduce precision"""
        grad_min = gradients.min()
        grad_max = gradients.max()
        grad_range = grad_max - grad_min

        if grad_range > 0:
            quantized = torch.round(
                (gradients - grad_min) / grad_range * (self.quantization_levels - 1)
            )
            quantized = (
                quantized / (self.quantization_levels - 1) * grad_range + grad_min
            )
        else:
            quantized = gradients

        return quantized

    def compress(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply compression (sparsification + quantization)"""
        sparse_grads, _ = self.sparsify_top_k(gradients)
        compressed_grads = self.quantize(sparse_grads)
        return compressed_grads
class FederatedRLClient:
    """Individual client in federated reinforcement learning"""

    def __init__(
        self,
        client_id: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        local_epochs: int = 5,
        privacy_epsilon: float = 1.0,
    ):

        self.client_id = client_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.local_epochs = local_epochs

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.privacy_engine = DifferentialPrivacy(epsilon=privacy_epsilon)
        self.compression = GradientCompression(compression_ratio=0.2)

        self.replay_buffer = deque(maxlen=10000)

        self.local_rewards = []
        self.communication_costs = []

    def collect_experience(self, env, n_episodes: int = 10):
        """Collect experience from local environment"""
        episode_rewards = []

        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_data = []

            for step in range(200):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                with torch.no_grad():
                    action = self.actor(state_tensor).squeeze().numpy()
                    value = self.critic(state_tensor).squeeze().item()

                action += np.random.normal(0, 0.1, action.shape)
                action = np.clip(action, -1, 1)

                next_state, reward, done, _ = env.step(action)

                episode_data.append(
                    {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "value": value,
                        "done": done,
                    }
                )

                episode_reward += reward
                state = next_state

                if done:
                    break

            episode_data = self._compute_advantages(episode_data)

            self.replay_buffer.extend(episode_data)
            episode_rewards.append(episode_reward)

        self.local_rewards.extend(episode_rewards)
        return np.mean(episode_rewards)

    def _compute_advantages(
        self, episode_data: List[Dict], gamma: float = 0.99, lambda_gae: float = 0.95
    ) -> List[Dict]:
        """Compute GAE advantages"""
        advantages = []
        gae = 0

        for t in reversed(range(len(episode_data))):
            if t == len(episode_data) - 1:
                next_value = 0
            else:
                next_value = episode_data[t + 1]["value"]

            delta = (
                episode_data[t]["reward"]
                + gamma * next_value * (1 - episode_data[t]["done"])
                - episode_data[t]["value"]
            )

            gae = delta + gamma * lambda_gae * (1 - episode_data[t]["done"]) * gae
            advantages.insert(0, gae)

        for i, advantage in enumerate(advantages):
            episode_data[i]["advantage"] = advantage

        return episode_data

    def local_update(
        self, global_actor: nn.Module = None, global_critic: nn.Module = None
    ) -> Dict:
        """Perform local training updates"""

        if global_actor is not None:
            self.actor.load_state_dict(global_actor.state_dict())
        if global_critic is not None:
            self.critic.load_state_dict(global_critic.state_dict())

        if len(self.replay_buffer) < 32:
            return {"actor_loss": 0, "critic_loss": 0}

        total_actor_loss = 0
        total_critic_loss = 0

        for epoch in range(self.local_epochs):
            batch_size = min(32, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, batch_size)

            states = torch.FloatTensor([t["state"] for t in batch])
            actions = torch.FloatTensor([t["action"] for t in batch])
            rewards = torch.FloatTensor([t["reward"] for t in batch])
            next_states = torch.FloatTensor([t["next_state"] for t in batch])
            advantages = torch.FloatTensor([t["advantage"] for t in batch])
            values = torch.FloatTensor([t["value"] for t in batch])

            returns = advantages + values

            predicted_values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(predicted_values, returns.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            predicted_actions = self.actor(states)

            action_loss = F.mse_loss(predicted_actions, actions)
            actor_loss = (action_loss * advantages.detach()).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "actor_loss": total_actor_loss / self.local_epochs,
            "critic_loss": total_critic_loss / self.local_epochs,
        }

    def get_model_updates(
        self, global_actor: nn.Module, global_critic: nn.Module
    ) -> Dict:
        """Get privatized and compressed model updates"""

        actor_updates = {}
        critic_updates = {}

        for (name, local_param), (_, global_param) in zip(
            self.actor.named_parameters(), global_actor.named_parameters()
        ):
            update = local_param.data - global_param.data

            private_update = self.privacy_engine.privatize_gradients(update)

            compressed_update = self.compression.compress(private_update)

            actor_updates[name] = compressed_update

        for (name, local_param), (_, global_param) in zip(
            self.critic.named_parameters(), global_critic.named_parameters()
        ):
            update = local_param.data - global_param.data
            private_update = self.privacy_engine.privatize_gradients(update)
            compressed_update = self.compression.compress(private_update)
            critic_updates[name] = compressed_update

        comm_cost = sum(u.numel() for u in actor_updates.values())
        comm_cost += sum(u.numel() for u in critic_updates.values())
        self.communication_costs.append(comm_cost)

        return {
            "actor_updates": actor_updates,
            "critic_updates": critic_updates,
            "num_samples": len(self.replay_buffer),
            "client_id": self.client_id,
        }
class FederatedRLServer:
    """Central server for federated reinforcement learning"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        aggregation_method: str = "fedavg",
        byzantine_tolerance: bool = False,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.aggregation_method = aggregation_method
        self.byzantine_tolerance = byzantine_tolerance

        self.global_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.global_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.round_statistics = []
        self.client_contributions = defaultdict(list)

    def aggregate_updates(self, selected_clients: np.ndarray, reward: Dict) -> Dict:
        """Aggregate updates from selected clients"""

        global_loss = reward.get("global_loss", 0.0)
        with torch.no_grad():
            for param in self.global_actor.parameters():
                param.data += (
                    0.01 * torch.randn_like(param) * (1.0 / (1.0 + global_loss))
                )
            for param in self.global_critic.parameters():
                param.data += (
                    0.01 * torch.randn_like(param) * (1.0 / (1.0 + global_loss))
                )

        return {
            "success": True,
            "num_selected_clients": len(selected_clients),
            "global_loss": global_loss,
        }