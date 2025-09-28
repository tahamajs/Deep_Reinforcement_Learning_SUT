"""
Distributed Reinforcement Learning

This module provides distributed computing approaches for reinforcement learning,
including parameter servers, worker nodes, and distributed training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import threading
import queue
import time
from collections import deque
import copy


class ParameterServer:
    """
    Parameter Server for distributed RL training.

    Manages global model parameters and coordinates updates from worker nodes.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        lr: float = 1e-3,
        **optimizer_kwargs,
    ):
        self.model = model
        self.global_model = copy.deepcopy(model)

        # Optimizer for global model
        self.optimizer = optimizer_class(
            self.global_model.parameters(), lr=lr, **optimizer_kwargs
        )

        # Communication queues
        self.gradient_queue = queue.Queue()
        self.parameter_queue = queue.Queue()
        self.request_queue = queue.Queue()

        # Worker management
        self.active_workers = set()
        self.worker_updates = {}

        # Statistics
        self.total_updates = 0
        self.update_times = deque(maxlen=100)

        # Control flags
        self.running = False
        self.server_thread = None

    def start(self):
        """Start parameter server."""
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop)
        self.server_thread.daemon = True
        self.server_thread.start()

    def stop(self):
        """Stop parameter server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)

    def _server_loop(self):
        """Main server loop."""
        while self.running:
            try:
                # Process gradient updates
                if not self.gradient_queue.empty():
                    worker_id, gradients = self.gradient_queue.get_nowait()
                    self._process_gradients(worker_id, gradients)

                # Process parameter requests
                if not self.request_queue.empty():
                    worker_id = self.request_queue.get_nowait()
                    self._send_parameters(worker_id)

                time.sleep(0.001)  # Small delay to prevent busy waiting

            except Exception as e:
                print(f"Parameter server error: {e}")

    def _process_gradients(self, worker_id: int, gradients: List[torch.Tensor]):
        """Process gradients from worker."""
        start_time = time.time()

        # Store gradients
        self.worker_updates[worker_id] = gradients

        # Check if we have enough updates for aggregation
        if len(self.worker_updates) >= len(self.active_workers):
            self._aggregate_gradients()

        self.update_times.append(time.time() - start_time)

    def _aggregate_gradients(self):
        """Aggregate gradients from all workers."""
        if not self.worker_updates:
            return

        # Initialize aggregated gradients
        aggregated_grads = []
        for param in self.global_model.parameters():
            aggregated_grads.append(torch.zeros_like(param.data))

        # Sum gradients from all workers
        num_workers = len(self.worker_updates)
        for worker_grads in self.worker_updates.values():
            for i, grad in enumerate(worker_grads):
                aggregated_grads[i] += grad / num_workers

        # Apply gradients
        self.optimizer.zero_grad()
        for param, grad in zip(self.global_model.parameters(), aggregated_grads):
            param.grad = grad

        self.optimizer.step()

        # Clear worker updates
        self.worker_updates.clear()
        self.total_updates += 1

    def _send_parameters(self, worker_id: int):
        """Send current parameters to worker."""
        params = [param.data.clone() for param in self.global_model.parameters()]
        self.parameter_queue.put((worker_id, params))

    def register_worker(self, worker_id: int):
        """Register a worker with the parameter server."""
        self.active_workers.add(worker_id)

    def unregister_worker(self, worker_id: int):
        """Unregister a worker."""
        self.active_workers.discard(worker_id)
        self.worker_updates.pop(worker_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get parameter server statistics."""
        return {
            "total_updates": self.total_updates,
            "active_workers": len(self.active_workers),
            "avg_update_time": np.mean(self.update_times) if self.update_times else 0.0,
            "parameter_count": sum(p.numel() for p in self.global_model.parameters()),
        }


class WorkerNode:
    """
    Worker Node for distributed RL training.

    Collects experience and computes gradients for the parameter server.
    """

    def __init__(
        self,
        worker_id: int,
        model: nn.Module,
        optimizer_class: type = torch.optim.Adam,
        lr: float = 1e-4,
        **optimizer_kwargs,
    ):
        self.worker_id = worker_id
        self.model = model
        self.local_model = copy.deepcopy(model)

        # Local optimizer
        self.optimizer = optimizer_class(
            self.local_model.parameters(), lr=lr, **optimizer_kwargs
        )

        # Communication queues (shared with parameter server)
        self.gradient_queue = None
        self.parameter_queue = None
        self.request_queue = None

        # Experience buffer
        self.experience_buffer = deque(maxlen=1000)

        # Statistics
        self.episodes_completed = 0
        self.total_steps = 0
        self.gradients_sent = 0

        # Control flags
        self.running = False
        self.worker_thread = None

    def connect_to_server(
        self,
        gradient_queue: queue.Queue,
        parameter_queue: queue.Queue,
        request_queue: queue.Queue,
    ):
        """Connect to parameter server."""
        self.gradient_queue = gradient_queue
        self.parameter_queue = parameter_queue
        self.request_queue = request_queue

    def start(self):
        """Start worker."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop(self):
        """Stop worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            try:
                # Request latest parameters
                self.request_queue.put(self.worker_id)

                # Wait for parameters
                try:
                    worker_id, params = self.parameter_queue.get(timeout=1.0)
                    if worker_id == self.worker_id:
                        self._update_local_model(params)
                except queue.Empty:
                    continue

                # Collect experience and compute gradients
                if len(self.experience_buffer) >= 32:  # Mini-batch size
                    gradients = self._compute_gradients()
                    if gradients:
                        self.gradient_queue.put((self.worker_id, gradients))
                        self.gradients_sent += 1

                time.sleep(0.01)  # Small delay

            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")

    def _update_local_model(self, params: List[torch.Tensor]):
        """Update local model with global parameters."""
        with torch.no_grad():
            for local_param, global_param in zip(self.local_model.parameters(), params):
                local_param.data.copy_(global_param)

    def _compute_gradients(self) -> Optional[List[torch.Tensor]]:
        """Compute gradients from experience buffer."""
        if not self.experience_buffer:
            return None

        # Sample mini-batch
        batch_size = min(32, len(self.experience_buffer))
        batch_indices = np.random.choice(
            len(self.experience_buffer), batch_size, replace=False
        )
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        # Compute gradients (simplified PPO-style)
        with torch.no_grad():
            next_values = self.local_model(next_states)[1]  # Value function
            targets = rewards + 0.99 * next_values * (1 - dones)

        values = self.local_model(states)[1]
        advantages = targets - values

        # Compute policy loss
        logits, _ = self.local_model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Simplified surrogate loss
        policy_loss = -(advantages * action_log_probs).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), targets.squeeze())

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        gradients = [param.grad.clone() for param in self.local_model.parameters()]

        return gradients

    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add experience to buffer."""
        self.experience_buffer.append((state, action, reward, next_state, done))
        self.total_steps += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "episodes_completed": self.episodes_completed,
            "total_steps": self.total_steps,
            "gradients_sent": self.gradients_sent,
            "buffer_size": len(self.experience_buffer),
        }


class DistributedRLTrainer:
    """
    Distributed RL Trainer coordinating parameter server and workers.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        num_workers: int = 4,
        optimizer_class: type = torch.optim.Adam,
        global_lr: float = 1e-3,
        local_lr: float = 1e-4,
    ):
        self.model_factory = model_factory
        self.num_workers = num_workers

        # Create global model
        self.global_model = model_factory()

        # Create parameter server
        self.param_server = ParameterServer(
            self.global_model, optimizer_class, global_lr
        )

        # Create workers
        self.workers = []
        for i in range(num_workers):
            worker_model = model_factory()
            worker = WorkerNode(i, worker_model, optimizer_class, local_lr)
            worker.connect_to_server(
                self.param_server.gradient_queue,
                self.param_server.parameter_queue,
                self.param_server.request_queue,
            )
            self.workers.append(worker)

        # Register workers
        for worker in self.workers:
            self.param_server.register_worker(worker.worker_id)

    def start_training(self):
        """Start distributed training."""
        print(f"Starting distributed training with {self.num_workers} workers")

        # Start parameter server
        self.param_server.start()

        # Start workers
        for worker in self.workers:
            worker.start()

    def stop_training(self):
        """Stop distributed training."""
        print("Stopping distributed training")

        # Stop workers
        for worker in self.workers:
            worker.stop()

        # Stop parameter server
        self.param_server.stop()

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        server_stats = self.param_server.get_stats()

        worker_stats = []
        for worker in self.workers:
            worker_stats.append(worker.get_stats())

        return {
            "server": server_stats,
            "workers": worker_stats,
            "total_workers": self.num_workers,
            "avg_worker_steps": np.mean([w["total_steps"] for w in worker_stats]),
            "total_gradients_sent": sum(w["gradients_sent"] for w in worker_stats),
        }

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "global_model": self.global_model.state_dict(),
            "param_server_optimizer": self.param_server.optimizer.state_dict(),
            "training_stats": self.get_training_stats(),
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)

        self.global_model.load_state_dict(checkpoint["global_model"])
        self.param_server.optimizer.load_state_dict(
            checkpoint["param_server_optimizer"]
        )

        # Update worker models
        for worker in self.workers:
            worker._update_local_model(
                [param.data for param in self.global_model.parameters()]
            )

    def simulate_worker_experience(
        self, env_factory: Callable, episodes_per_worker: int = 10
    ):
        """
        Simulate experience collection for workers.

        Args:
            env_factory: Function that creates environment instances
            episodes_per_worker: Number of episodes per worker
        """

        def worker_simulation(worker_id: int):
            worker = self.workers[worker_id]
            env = env_factory()

            for episode in range(episodes_per_worker):
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32)
                done = False
                episode_steps = 0

                while not done and episode_steps < 1000:
                    # Get action from local model
                    with torch.no_grad():
                        action, _ = worker.local_model.get_action(state.unsqueeze(0))
                        action = action.item()

                    # Execute action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Convert to tensors
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    reward = torch.tensor(reward, dtype=torch.float32)
                    done_tensor = torch.tensor(done, dtype=torch.float32)

                    # Add experience
                    worker.add_experience(
                        state, torch.tensor(action), reward, next_state, done_tensor
                    )

                    state = next_state
                    episode_steps += 1

                worker.episodes_completed += 1

        # Start simulation threads
        threads = []
        for i in range(self.num_workers):
            thread = threading.Thread(target=worker_simulation, args=(i,))
            thread.daemon = True
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()
