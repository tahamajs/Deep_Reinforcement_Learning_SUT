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
        state_dim: int = None,
        action_dim: int = None,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        local_epochs: int = 5,
        privacy_epsilon: float = 1.0,
        local_model: nn.Module = None,
        environment=None,
        learning_rate: float = None,
        use_differential_privacy: bool = False,
        clip_norm: float = 1.0,
        compression_rate: float = 1.0,
    ):

        self.client_id = client_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.local_epochs = local_epochs
        self.local_model = local_model
        self.environment = environment
        self.learning_rate = learning_rate if learning_rate is not None else lr
        self.use_differential_privacy = use_differential_privacy
        self.clip_norm = clip_norm
        self.compression_rate = compression_rate

        if local_model is not None:
            # Use provided local model
            self.local_model = local_model
        elif state_dim is not None and action_dim is not None:
            # Create default actor-critic models
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

        if hasattr(self, "actor"):
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        if hasattr(self, "critic"):
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

            for step in range(200):  # Max episode length
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

    def download_model(
        self, global_model: Union[nn.Module, Tuple[nn.Module, nn.Module]]
    ):
        """Download global model for demo compatibility"""
        if hasattr(self, "local_model") and self.local_model is not None:
            self.local_model.load_state_dict(global_model.state_dict())
        # Also update actor/critic if they exist
        if isinstance(global_model, tuple):
            global_actor, global_critic = global_model
            self.actor.load_state_dict(global_actor.state_dict())
            self.critic.load_state_dict(global_critic.state_dict())

    def local_training(self, n_episodes: int = 10, n_epochs: int = 5) -> List[float]:
        """Local training for demo compatibility"""
        if hasattr(self, "environment") and self.environment is not None:
            rewards = []
            for _ in range(n_episodes):
                state = self.environment.reset()
                episode_reward = 0
                done = False

                while not done:
                    # Simple action selection
                    with torch.no_grad():
                        if (
                            hasattr(self, "local_model")
                            and self.local_model is not None
                        ):
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            action_probs = self.local_model(state_tensor)
                            action = torch.argmax(action_probs, dim=1).item()
                        else:
                            action = np.random.choice(self.action_dim)

                    next_state, reward, done, _ = self.environment.step(action)
                    episode_reward += reward
                    state = next_state

                rewards.append(episode_reward)

            return rewards
        else:
            # Fallback to existing train_local method
            return [self.train_local([], 1e-3, gamma=0.99) for _ in range(n_episodes)]

    def get_model_update(self) -> Dict:
        """Get model update for demo compatibility"""
        if hasattr(self, "local_model") and self.local_model is not None:
            return {"model_state": self.local_model.state_dict()}
        else:
            return self.get_model_updates(self.actor, self.critic)

    def get_communication_cost(self) -> float:
        """Get communication cost for demo"""
        # Estimate based on model size and compression
        if hasattr(self, "local_model") and self.local_model is not None:
            n_params = sum(p.numel() for p in self.local_model.parameters())
        else:
            n_params = sum(p.numel() for p in self.actor.parameters())
            n_params += sum(p.numel() for p in self.critic.parameters())

        # Assume 4 bytes per parameter, convert to MB
        cost_mb = (n_params * 4) / (1024 * 1024)

        # Apply compression if set
        if hasattr(self, "compression_rate"):
            cost_mb *= self.compression_rate

        return cost_mb


class FederatedRLServer:
    """Central server for federated reinforcement learning"""

    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        hidden_dim: int = 64,
        aggregation_method: str = "fedavg",
        byzantine_tolerance: bool = False,
        global_model: nn.Module = None,
        n_clients: int = None,
        use_differential_privacy: bool = False,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        compression_rate: float = 1.0,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.aggregation_method = aggregation_method
        self.byzantine_tolerance = byzantine_tolerance
        self.n_clients = n_clients
        self.use_differential_privacy = use_differential_privacy
        self.epsilon = epsilon
        self.delta = delta
        self.compression_rate = compression_rate
        self.global_model = global_model

        if global_model is not None:
            # Use provided global model
            self.global_model = global_model
        elif state_dim is not None and action_dim is not None:
            # Create default actor-critic models
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

    def aggregate_updates(self, client_updates: List[Dict]) -> Dict:
        """Aggregate client updates using specified method"""

        if len(client_updates) == 0:
            return {"success": False, "message": "No client updates"}

        if self.aggregation_method == "fedavg":
            return self._fedavg_aggregation(client_updates)
        elif self.aggregation_method == "fedprox":
            return self._fedprox_aggregation(client_updates)
        elif self.aggregation_method == "trimmed_mean":
            return self._trimmed_mean_aggregation(client_updates)
        else:
            return self._fedavg_aggregation(client_updates)

    def _fedavg_aggregation(self, client_updates: List[Dict]) -> Dict:
        """FedAvg aggregation with weighted averaging"""

        total_samples = sum(update["num_samples"] for update in client_updates)
        weights = [update["num_samples"] / total_samples for update in client_updates]

        aggregated_actor_updates = {}
        for name, param in self.global_actor.named_parameters():
            weighted_updates = []
            for i, update in enumerate(client_updates):
                if name in update["actor_updates"]:
                    weighted_updates.append(weights[i] * update["actor_updates"][name])

            if weighted_updates:
                aggregated_actor_updates[name] = torch.stack(weighted_updates).sum(
                    dim=0
                )

        aggregated_critic_updates = {}
        for name, param in self.global_critic.named_parameters():
            weighted_updates = []
            for i, update in enumerate(client_updates):
                if name in update["critic_updates"]:
                    weighted_updates.append(weights[i] * update["critic_updates"][name])

            if weighted_updates:
                aggregated_critic_updates[name] = torch.stack(weighted_updates).sum(
                    dim=0
                )

        with torch.no_grad():
            for name, param in self.global_actor.named_parameters():
                if name in aggregated_actor_updates:
                    param.data += aggregated_actor_updates[name]

            for name, param in self.global_critic.named_parameters():
                if name in aggregated_critic_updates:
                    param.data += aggregated_critic_updates[name]

        return {
            "success": True,
            "aggregation_method": "fedavg",
            "num_clients": len(client_updates),
            "total_samples": total_samples,
        }

    def _trimmed_mean_aggregation(self, client_updates: List[Dict]) -> Dict:
        """Byzantine-robust trimmed mean aggregation"""

        trim_ratio = 0.1  # Trim 10% from each side

        for name, param in self.global_actor.named_parameters():
            param_updates = []
            for update in client_updates:
                if name in update["actor_updates"]:
                    param_updates.append(update["actor_updates"][name])

            if param_updates:
                stacked_updates = torch.stack(param_updates)
                trimmed_mean = self._compute_trimmed_mean(stacked_updates, trim_ratio)
                param.data += trimmed_mean

        for name, param in self.global_critic.named_parameters():
            param_updates = []
            for update in client_updates:
                if name in update["critic_updates"]:
                    param_updates.append(update["critic_updates"][name])

            if param_updates:
                stacked_updates = torch.stack(param_updates)
                trimmed_mean = self._compute_trimmed_mean(stacked_updates, trim_ratio)
                param.data += trimmed_mean

        return {
            "success": True,
            "aggregation_method": "trimmed_mean",
            "num_clients": len(client_updates),
        }

    def _compute_trimmed_mean(
        self, tensor_stack: torch.Tensor, trim_ratio: float
    ) -> torch.Tensor:
        """Compute trimmed mean along first dimension"""
        n_clients = tensor_stack.shape[0]
        n_trim = int(n_clients * trim_ratio)

        if n_trim == 0:
            return tensor_stack.mean(dim=0)

        sorted_tensor, _ = torch.sort(tensor_stack, dim=0)

        trimmed_tensor = sorted_tensor[n_trim:-n_trim] if n_trim > 0 else sorted_tensor
        return trimmed_tensor.mean(dim=0)

    def _fedprox_aggregation(self, client_updates: List[Dict]) -> Dict:
        """FedProx aggregation (simplified version)"""
        return self._fedavg_aggregation(client_updates)

    def evaluate_global_model(self, test_env) -> float:
        """Evaluate global model performance"""

        total_reward = 0
        n_episodes = 10

        for episode in range(n_episodes):
            state = test_env.reset()
            episode_reward = 0

            for step in range(200):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                with torch.no_grad():
                    action = self.global_actor(state_tensor).squeeze().numpy()

                next_state, reward, done, _ = test_env.step(action)
                episode_reward += reward
                state = next_state

                if done:
                    break

            total_reward += episode_reward

        return total_reward / n_episodes

    def get_global_models(self) -> Tuple[nn.Module, nn.Module]:
        """Get copies of global models"""
        global_actor_copy = copy.deepcopy(self.global_actor)
        global_critic_copy = copy.deepcopy(self.global_critic)
        return global_actor_copy, global_critic_copy

    def get_global_model(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        """Get global model for demo compatibility"""
        if hasattr(self, "global_model") and self.global_model is not None:
            return copy.deepcopy(self.global_model)
        else:
            return self.get_global_models()

    def get_noise_scale(self) -> float:
        """Get noise scale for privacy metrics"""
        if hasattr(self, "use_differential_privacy") and self.use_differential_privacy:
            # Simple noise scale calculation
            return 2.0 * self.epsilon if hasattr(self, "epsilon") else 1.0
        return 0.0


# Add aliases for compatibility with demo
FederatedAgent = FederatedRLClient
FederatedServer = FederatedRLServer


class SimpleAgent(nn.Module):
    """Simple neural network agent for federated RL demos"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(SimpleAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


class FederatedEnvironment:
    """Simple environment for federated RL demonstrations"""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        reward_bias: float = 0.0,
        transition_noise: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_bias = reward_bias
        self.transition_noise = transition_noise
        self.state = None
        self.step_count = 0
        self.max_steps = 50

    def reset(self):
        """Reset environment to initial state"""
        self.state = np.random.randn(self.state_dim) * 0.1
        self.step_count = 0
        return self.state

    def step(self, action: int):
        """Take an action and return next state, reward, done, info"""
        # Simple dynamics
        action_effect = np.zeros(self.state_dim)
        action_effect[action % self.state_dim] = 0.1

        # Update state with action effect and noise
        self.state = (
            self.state
            + action_effect
            + np.random.randn(self.state_dim) * self.transition_noise
        )
        self.state = np.clip(self.state, -2, 2)

        # Calculate reward (distance-based with client-specific bias)
        distance = np.linalg.norm(self.state)
        reward = -distance + self.reward_bias + np.random.randn() * 0.1

        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps or distance > 1.5

        return self.state, reward, done, {}
