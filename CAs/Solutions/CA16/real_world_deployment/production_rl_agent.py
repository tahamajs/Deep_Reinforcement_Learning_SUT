"""
Production RL Agent

This module provides production-ready RL agents with serving capabilities,
load balancing, and performance monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import threading
import time
import queue
from collections import deque
import logging
import json
from datetime import datetime


class ProductionRLAgent(nn.Module):
    """
    Production-ready RL agent with monitoring and safety features.

    Includes model serving, performance tracking, and safety constraints.
    """

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: Optional[nn.Module] = None,
        device: str = "cpu",
        safety_threshold: float = 0.8,
    ):
        super().__init__()

        self.policy_net = policy_net
        self.value_net = value_net or policy_net  # Use same network if not provided
        self.device = device
        self.safety_threshold = safety_threshold

        self.to(device)
        self.eval()  # Production mode

        self.request_count = 0
        self.inference_times = deque(maxlen=1000)
        self.action_distribution = {}
        self.error_count = 0

        self.safety_violations = 0
        self.last_safety_check = time.time()

        self.logger = logging.getLogger("ProductionRLAgent")
        self.logger.setLevel(logging.INFO)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with performance monitoring.

        Args:
            state: Input state

        Returns:
            Action logits and value estimate
        """
        start_time = time.time()

        try:
            with torch.no_grad():
                state = state.to(self.device)

                if hasattr(self.policy_net, "forward"):
                    policy_output = self.policy_net(state)
                else:
                    policy_output = self.policy_net(state)

                if self.value_net != self.policy_net:
                    value_output = self.value_net(state)
                else:
                    if isinstance(policy_output, tuple):
                        policy_output, value_output = policy_output
                    else:
                        value_output = torch.zeros(state.size(0), 1, device=self.device)

                if policy_output.dim() == 1:
                    policy_output = policy_output.unsqueeze(0)
                if value_output.dim() == 1:
                    value_output = value_output.unsqueeze(0)

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Inference error: {e}")
            batch_size = state.size(0) if state.dim() > 1 else 1
            policy_output = torch.zeros(
                batch_size, self.get_action_dim(), device=self.device
            )
            value_output = torch.zeros(batch_size, 1, device=self.device)

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.request_count += 1

        return policy_output, value_output

    def get_action(
        self, state: torch.Tensor, deterministic: bool = True, safety_check: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get action with safety checking and monitoring.

        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            safety_check: Whether to perform safety validation

        Returns:
            Action and metadata
        """
        logits, value = self.forward(state)

        if safety_check:
            is_safe, safety_info = self._check_safety(state, logits)
            if not is_safe:
                self.safety_violations += 1
                self.logger.warning(f"Safety violation: {safety_info}")
                safe_action = self._get_safe_action(state)
                return safe_action, {
                    "safety_violation": True,
                    "original_action": torch.argmax(logits, dim=-1),
                    "safety_info": safety_info,
                }

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        action_idx = action.item() if action.dim() == 0 else action[0].item()
        self.action_distribution[action_idx] = (
            self.action_distribution.get(action_idx, 0) + 1
        )

        metadata = {
            "logits": logits.detach().cpu().numpy(),
            "value": value.item() if value.dim() == 0 else value[0].item(),
            "inference_time": self.inference_times[-1] if self.inference_times else 0.0,
            "safety_violation": False,
        }

        return action, metadata

    def _check_safety(
        self, state: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Check if action is safe.

        Args:
            state: Current state
            logits: Action logits

        Returns:
            (is_safe, violation_reason)
        """
        max_prob = torch.softmax(logits, dim=-1).max().item()

        if max_prob < self.safety_threshold:
            return False, f"Low confidence action (prob={max_prob:.3f})"


        return True, ""

    def _get_safe_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get a safe fallback action."""
        return torch.tensor(0, device=self.device)  # Default safe action

    def get_action_dim(self) -> int:
        """Get action dimension."""
        try:
            dummy_input = torch.zeros(1, self.get_state_dim(), device=self.device)
            logits, _ = self.forward(dummy_input)
            return logits.size(-1)
        except:
            return 1  # Default

    def get_state_dim(self) -> int:
        """Get state dimension."""
        return getattr(self, "_state_dim", 1)

    def set_state_dim(self, dim: int):
        """Set state dimension."""
        self._state_dim = dim

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_requests": self.request_count,
            "average_inference_time": (
                np.mean(self.inference_times) if self.inference_times else 0.0
            ),
            "p95_inference_time": (
                np.percentile(self.inference_times, 95) if self.inference_times else 0.0
            ),
            "error_rate": self.error_count / max(1, self.request_count),
            "safety_violations": self.safety_violations,
            "action_distribution": self.action_distribution.copy(),
            "uptime": time.time() - getattr(self, "_start_time", time.time()),
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.request_count = 0
        self.inference_times.clear()
        self.action_distribution.clear()
        self.error_count = 0
        self.safety_violations = 0
        self._start_time = time.time()


class ModelServing:
    """
    Model serving system for production RL agents.

    Handles model loading, versioning, and serving requests.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.current_model = None
        self.model_version = None

        self.model_versions = {}

        self.total_requests = 0
        self.active_models = 0

    def load_model(self, version: str = "latest") -> ProductionRLAgent:
        """
        Load a specific model version.

        Args:
            version: Model version to load

        Returns:
            Loaded model
        """
        if version == "latest":
            version = max(self.model_versions.keys()) if self.model_versions else "v1.0"

        model_path = self.model_versions.get(version, self.model_path)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            model = checkpoint.get("model")
            if model is None:
                raise ValueError("Model not found in checkpoint")

            agent = ProductionRLAgent(model, device=self.device)
            agent.model_version = version

            self.current_model = agent
            self.model_version = version
            self.active_models += 1

            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to load model {version}: {e}")

    def register_model_version(self, version: str, path: str):
        """Register a model version."""
        self.model_versions[version] = path

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "current_version": self.model_version,
            "available_versions": list(self.model_versions.keys()),
            "active_models": self.active_models,
            "total_requests": self.total_requests,
        }

    def serve_request(self, state: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Serve a single inference request.

        Args:
            state: Input state
            **kwargs: Additional arguments for get_action

        Returns:
            Inference result
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded")

        self.total_requests += 1

        try:
            action, metadata = self.current_model.get_action(state, **kwargs)
            return {
                "action": action.item() if action.dim() == 0 else action.tolist(),
                "metadata": metadata,
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}


class LoadBalancer:
    """
    Load balancer for distributing requests across multiple model instances.
    """

    def __init__(
        self, model_paths: List[str], num_instances: int = 2, device: str = "cpu"
    ):
        self.model_paths = model_paths
        self.num_instances = num_instances
        self.device = device

        self.instances = []
        self.instance_loads = []

        for i in range(num_instances):
            serving = ModelServing(model_paths[i % len(model_paths)], device)
            model = serving.load_model()
            self.instances.append(model)
            self.instance_loads.append(0)

        self.request_queue = queue.Queue()
        self.response_queues = {}

        self.workers = []
        self.running = False

    def start(self):
        """Start load balancer."""
        self.running = True

        for i in range(self.num_instances):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def stop(self):
        """Stop load balancer."""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)

    def submit_request(self, request_id: str, state: torch.Tensor, **kwargs) -> str:
        """
        Submit a request for processing.

        Args:
            request_id: Unique request identifier
            state: Input state
            **kwargs: Additional arguments

        Returns:
            Request ID for tracking
        """
        self.response_queues[request_id] = queue.Queue()
        self.request_queue.put((request_id, state, kwargs))

        return request_id

    def get_response(
        self, request_id: str, timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get response for a request.

        Args:
            request_id: Request identifier
            timeout: Timeout in seconds

        Returns:
            Response or None if timeout
        """
        try:
            response_queue = self.response_queues[request_id]
            response = response_queue.get(timeout=timeout)
            del self.response_queues[request_id]
            return response
        except (queue.Empty, KeyError):
            return None

    def _worker_loop(self, instance_id: int):
        """Worker loop for processing requests."""
        instance = self.instances[instance_id]

        while self.running:
            try:
                request_id, state, kwargs = self.request_queue.get(timeout=1.0)

                self.instance_loads[instance_id] += 1
                response = instance.get_action(state, **kwargs)
                self.instance_loads[instance_id] -= 1

                if request_id in self.response_queues:
                    self.response_queues[request_id].put(
                        {
                            "action": response[0],
                            "metadata": response[1],
                            "instance_id": instance_id,
                        }
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {instance_id} error: {e}")

    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            "instance_loads": self.instance_loads.copy(),
            "total_load": sum(self.instance_loads),
            "queue_size": self.request_queue.qsize(),
            "active_requests": len(self.response_queues),
        }
