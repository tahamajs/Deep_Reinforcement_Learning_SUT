import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque, defaultdict
import copy
import random
import time
import json
import os
from pathlib import Path

# Advanced Replay Buffer with Prioritization and Quantum States
class QuantumPrioritizedReplayBuffer:
    """Prioritized replay buffer with quantum-inspired state representation"""

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        quantum_dim: int = 8,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.quantum_dim = quantum_dim

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

        # Quantum state tracking for importance sampling
        self.quantum_states = np.zeros((capacity, quantum_dim), dtype=np.complex64)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        quantum_state: Optional[np.ndarray] = None,
    ):
        """Add experience to buffer with quantum state"""

        if quantum_state is None:
            # Generate quantum state from classical state
            quantum_state = self._classical_to_quantum(state)

        experience = (state, action, reward, next_state, done, quantum_state)

        if self.size < self.capacity:
            self.buffer.append(experience)
            self.quantum_states[self.size] = quantum_state
            self.priorities[self.size] = 1.0
            self.size += 1
        else:
            self.buffer[self.position] = experience
            self.quantum_states[self.position] = quantum_state
            self.priorities[self.position] = 1.0

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with prioritized quantum-aware sampling"""

        if self.size < batch_size:
            return None

        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Extract batch
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, quantum_states = zip(*batch)

        return {
            'states': torch.FloatTensor(np.array(states)),
            'actions': torch.FloatTensor(np.array(actions)),
            'rewards': torch.FloatTensor(rewards),
            'next_states': torch.FloatTensor(np.array(next_states)),
            'dones': torch.FloatTensor(dones),
            'quantum_states': torch.complex64(torch.from_numpy(np.array(quantum_states))),
            'weights': torch.FloatTensor(weights),
            'indices': indices,
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero

    def _classical_to_quantum(self, state: np.ndarray) -> np.ndarray:
        """Convert classical state to quantum representation"""
        # Simple amplitude encoding
        normalized_state = state / (np.linalg.norm(state) + 1e-6)
        quantum_state = np.zeros(self.quantum_dim, dtype=np.complex64)

        # Map to quantum amplitudes
        for i in range(min(len(normalized_state), self.quantum_dim)):
            quantum_state[i] = normalized_state[i] + 1j * normalized_state[i] * 0.1

        # Normalize quantum state
        quantum_state /= np.linalg.norm(quantum_state)

        return quantum_state

    def __len__(self):
        return self.size

# Quantum-Inspired Metrics Tracker
class QuantumMetricsTracker:
    """Advanced metrics tracking with quantum uncertainty quantification"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.quantum_uncertainty = defaultdict(float)

        # Quantum state for uncertainty tracking
        self.quantum_dim = 16
        self.quantum_state = np.random.uniform(0, 2*np.pi, self.quantum_dim)

    def update(self, metric_name: str, value: float, uncertainty: Optional[float] = None):
        """Update metric with quantum uncertainty tracking"""

        self.metrics[metric_name].append(value)

        # Compute quantum uncertainty if not provided
        if uncertainty is None:
            uncertainty = self._compute_quantum_uncertainty(value)

        self.quantum_uncertainty[metric_name] = uncertainty

        # Update quantum state based on metric
        phase_shift = value * 0.1 + uncertainty * 0.05
        self.quantum_state = (self.quantum_state + phase_shift) % (2 * np.pi)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get comprehensive statistics for metric"""

        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return {}

        values = list(self.metrics[metric_name])
        uncertainty = self.quantum_uncertainty[metric_name]

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'quantum_uncertainty': uncertainty,
            'coherence': self._compute_coherence(),
            'stability': 1.0 / (1.0 + np.var(values)),
        }

    def _compute_quantum_uncertainty(self, value: float) -> float:
        """Compute quantum-inspired uncertainty measure"""
        # Use quantum state phase differences
        phase_diffs = np.diff(self.quantum_state)
        uncertainty = np.mean(np.abs(phase_diffs)) / np.pi
        return min(uncertainty + abs(value) * 0.01, 1.0)

    def _compute_coherence(self) -> float:
        """Compute quantum coherence measure"""
        # Measure how coherent the quantum state is
        coherence = 1.0 / (1.0 + np.var(self.quantum_state))
        return coherence

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot all tracked metrics"""

        if not self.metrics:
            return

        n_metrics = len(self.metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for i, (metric_name, values) in enumerate(self.metrics.items()):
            ax = axes[i]
            values_list = list(values)

            ax.plot(values_list, label=metric_name, alpha=0.7)
            ax.set_title(f'{metric_name} (Quantum Uncertainty: {self.quantum_uncertainty[metric_name]:.3f})')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

# Advanced Logger with Quantum State Persistence
class QuantumLogger:
    """Advanced logging with quantum state persistence and uncertainty tracking"""

    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.current_episode = 0
        self.quantum_state_history = []

        # Quantum state for logging coherence
        self.quantum_dim = 8
        self.quantum_state = np.random.uniform(0, 2*np.pi, self.quantum_dim)

    def log_episode(self, episode_data: Dict[str, Any]):
        """Log episode data with quantum state"""

        # Add quantum coherence to episode data
        coherence = self._compute_coherence()
        episode_data['quantum_coherence'] = coherence
        episode_data['episode'] = self.current_episode
        episode_data['timestamp'] = time.time()

        # Save to JSONL file
        with open(self.metrics_file, 'a') as f:
            json.dump(episode_data, f)
            f.write('\n')

        # Update quantum state
        reward = episode_data.get('total_reward', 0)
        length = episode_data.get('episode_length', 1)
        phase_shift = (reward / length) * 0.1
        self.quantum_state = (self.quantum_state + phase_shift) % (2 * np.pi)

        self.quantum_state_history.append(self.quantum_state.copy())
        self.current_episode += 1

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       episode: int, additional_data: Optional[Dict] = None):
        """Save model checkpoint with quantum state"""

        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'quantum_state': self.quantum_state,
            'quantum_state_history': self.quantum_state_history[-100:],  # Last 100 states
            'timestamp': time.time(),
        }

        if additional_data:
            checkpoint.update(additional_data)

        checkpoint_path = self.checkpoints_dir / f"checkpoint_ep{episode}.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint with quantum state restoration"""

        checkpoint = torch.load(checkpoint_path)

        if 'quantum_state' in checkpoint:
            self.quantum_state = checkpoint['quantum_state']
            if 'quantum_state_history' in checkpoint:
                self.quantum_state_history = checkpoint['quantum_state_history']

        return checkpoint

    def _compute_coherence(self) -> float:
        """Compute quantum coherence"""
        return 1.0 / (1.0 + np.var(self.quantum_state))

    def get_experiment_summary(self) -> Dict:
        """Get comprehensive experiment summary"""

        if not self.metrics_file.exists():
            return {}

        episodes = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                episodes.append(json.loads(line))

        if not episodes:
            return {}

        # Compute statistics
        rewards = [ep.get('total_reward', 0) for ep in episodes]
        lengths = [ep.get('episode_length', 1) for ep in episodes]
        coherences = [ep.get('quantum_coherence', 0) for ep in episodes]

        return {
            'total_episodes': len(episodes),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'avg_episode_length': np.mean(lengths),
            'avg_quantum_coherence': np.mean(coherences),
            'experiment_duration': episodes[-1]['timestamp'] - episodes[0]['timestamp'],
        }

# Quantum-Inspired Random Number Generator
class QuantumRNG:
    """Quantum-inspired random number generator for exploration"""

    def __init__(self, seed: Optional[int] = None, quantum_dim: int = 32):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.quantum_dim = quantum_dim
        self.quantum_state = np.random.uniform(0, 2*np.pi, quantum_dim)
        self.phase_history = []

    def quantum_random(self, shape: Tuple = ()) -> np.ndarray:
        """Generate quantum-inspired random numbers"""

        # Evolve quantum state
        self.quantum_state += np.random.normal(0, 0.1, self.quantum_dim)
        self.quantum_state %= 2 * np.pi

        # Generate random numbers from quantum amplitudes
        amplitudes = np.exp(1j * self.quantum_state)
        probabilities = np.abs(amplitudes) ** 2
        probabilities /= np.sum(probabilities)

        if shape == ():
            # Single random number
            return np.random.choice(self.quantum_dim, p=probabilities) / self.quantum_dim
        else:
            # Array of random numbers
            indices = np.random.choice(self.quantum_dim, size=shape, p=probabilities)
            return indices.astype(float) / self.quantum_dim

    def quantum_choice(self, options: List[Any], size: Optional[int] = None) -> Union[Any, List[Any]]:
        """Make quantum-inspired choice from options"""

        n_options = len(options)
        quantum_probs = np.abs(np.exp(1j * self.quantum_state[:n_options])) ** 2
        quantum_probs /= np.sum(quantum_probs)

        if size is None:
            choice_idx = np.random.choice(n_options, p=quantum_probs)
            return options[choice_idx]
        else:
            choice_indices = np.random.choice(n_options, size=size, p=quantum_probs)
            return [options[i] for i in choice_indices]

    def get_coherence(self) -> float:
        """Get quantum coherence measure"""
        return 1.0 / (1.0 + np.var(self.quantum_state))

# Utility Functions
def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Soft update target network parameters"""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def hard_update(target_net: nn.Module, source_net: nn.Module):
    """Hard update target network parameters"""
    target_net.load_state_dict(source_net.state_dict())

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
               dones: torch.Tensor, gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)"""

    advantages = torch.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages[t] = gae

    return advantages

def normalize_tensor(tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize tensor to have zero mean and unit variance"""
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True) + epsilon
    return (tensor - mean) / std

def create_mlp_network(layer_sizes: List[int], activation: nn.Module = nn.ReLU,
                      output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    """Create multi-layer perceptron network"""

    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())

    return nn.Sequential(*layers)

def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, filepath: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_model_checkpoint(filepath: str, model: nn.Module,
                         optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint

print("✅ Advanced Utils implementations complete!")
print("Components implemented:")
print("- QuantumPrioritizedReplayBuffer: Prioritized replay with quantum states")
print("- QuantumMetricsTracker: Metrics tracking with quantum uncertainty")
print("- QuantumLogger: Advanced logging with quantum state persistence")
print("- QuantumRNG: Quantum-inspired random number generation")
print("- Utility functions: soft_update, hard_update, compute_gae, normalize_tensor, create_mlp_network, save/load checkpoints")