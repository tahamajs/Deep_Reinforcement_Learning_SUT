"""
Production-Ready RL Systems

This module contains classes for deploying RL systems in production environments,
including microservices architecture, deployment strategies, and performance monitoring.
"""

import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import threading
import queue
import psutil
import requests
from pathlib import Path


@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    latency: float
    throughput: float
    error_rate: float
    model_accuracy: float
    prediction_time: float
    queue_size: int


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""

    model_path: str
    version: str
    environment: str
    replicas: int
    resources: Dict[str, Any]
    health_check_interval: int
    scaling_threshold: float
    rollback_threshold: float


class PerformanceMonitor:
    """Monitor system performance and model metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.alerts = []
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "gpu_usage": 90.0,
            "latency": 1000.0,  # ms
            "error_rate": 0.05,
            "model_accuracy": 0.8,
        }

        # Performance tracking
        self.performance_history = {
            "latency": [],
            "throughput": [],
            "accuracy": [],
            "resource_usage": [],
        }

        # Alert system
        self.alert_callbacks = []

    def add_metrics(self, metrics: SystemMetrics):
        """Add new metrics to the monitor."""
        self.metrics_history.append(metrics)
        self._check_thresholds(metrics)
        self._update_performance_history(metrics)

    def _check_thresholds(self, metrics: SystemMetrics):
        """Check if metrics exceed thresholds and trigger alerts."""
        for metric_name, threshold in self.thresholds.items():
            value = getattr(metrics, metric_name, None)
            if value is not None and value > threshold:
                alert = {
                    "timestamp": metrics.timestamp,
                    "metric": metric_name,
                    "value": value,
                    "threshold": threshold,
                    "severity": "warning" if value < threshold * 1.5 else "critical",
                }
                self.alerts.append(alert)
                self._trigger_alert_callbacks(alert)

    def _update_performance_history(self, metrics: SystemMetrics):
        """Update performance history."""
        self.performance_history["latency"].append(metrics.latency)
        self.performance_history["throughput"].append(metrics.throughput)
        self.performance_history["accuracy"].append(metrics.model_accuracy)
        self.performance_history["resource_usage"].append(
            {
                "cpu": metrics.cpu_usage,
                "memory": metrics.memory_usage,
                "gpu": metrics.gpu_usage,
            }
        )

    def _trigger_alert_callbacks(self, alert: Dict[str, Any]):
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")

    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register an alert callback function."""
        self.alert_callbacks.append(callback)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}

        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics

        return {
            "avg_latency": np.mean([m.latency for m in recent_metrics]),
            "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
            "avg_accuracy": np.mean([m.model_accuracy for m in recent_metrics]),
            "avg_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "error_rate": np.mean([m.error_rate for m in recent_metrics]),
            "total_alerts": len(self.alerts),
            "recent_alerts": len(
                [a for a in self.alerts if a["timestamp"] > time.time() - 3600]
            ),
        }


class ModelDeploymentStrategy(ABC):
    """Abstract base class for model deployment strategies."""

    @abstractmethod
    def deploy(self, config: DeploymentConfig) -> bool:
        """Deploy the model with the given configuration."""
        pass

    @abstractmethod
    def rollback(self, version: str) -> bool:
        """Rollback to a previous version."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the deployment is healthy."""
        pass


class BlueGreenDeployment(ModelDeploymentStrategy):
    """Blue-Green deployment strategy."""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.active_environment = "blue"
        self.inactive_environment = "green"
        self.deployment_history = []

    def deploy(self, config: DeploymentConfig) -> bool:
        """Deploy using blue-green strategy."""
        try:
            # Deploy to inactive environment
            deployment_start = time.time()

            # Simulate deployment process
            time.sleep(2)  # Simulate deployment time

            # Switch traffic to new environment
            old_active = self.active_environment
            self.active_environment = self.inactive_environment
            self.inactive_environment = old_active

            deployment_time = time.time() - deployment_start

            # Record deployment
            deployment_record = {
                "timestamp": time.time(),
                "version": config.version,
                "environment": self.active_environment,
                "deployment_time": deployment_time,
                "status": "success",
            }
            self.deployment_history.append(deployment_record)

            return True

        except Exception as e:
            logging.error(f"Blue-green deployment failed: {e}")
            return False

    def rollback(self, version: str) -> bool:
        """Rollback to previous version."""
        try:
            # Switch back to previous environment
            old_active = self.active_environment
            self.active_environment = self.inactive_environment
            self.inactive_environment = old_active

            # Record rollback
            rollback_record = {
                "timestamp": time.time(),
                "version": version,
                "environment": self.active_environment,
                "status": "rollback",
            }
            self.deployment_history.append(rollback_record)

            return True

        except Exception as e:
            logging.error(f"Rollback failed: {e}")
            return False

    def health_check(self) -> bool:
        """Check if the active environment is healthy."""
        # Simulate health check
        recent_metrics = list(self.monitor.metrics_history)[-10:]
        if not recent_metrics:
            return True

        # Check if error rate is acceptable
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        return avg_error_rate < 0.1


class CanaryDeployment(ModelDeploymentStrategy):
    """Canary deployment strategy."""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.canary_traffic_percentage = 0.0
        self.canary_threshold = 0.1  # 10% error rate threshold
        self.deployment_history = []

    def deploy(self, config: DeploymentConfig) -> bool:
        """Deploy using canary strategy."""
        try:
            # Start with small percentage of traffic
            self.canary_traffic_percentage = 0.1  # 10%

            deployment_record = {
                "timestamp": time.time(),
                "version": config.version,
                "canary_percentage": self.canary_traffic_percentage,
                "status": "canary_started",
            }
            self.deployment_history.append(deployment_record)

            return True

        except Exception as e:
            logging.error(f"Canary deployment failed: {e}")
            return False

    def rollback(self, version: str) -> bool:
        """Rollback canary deployment."""
        try:
            self.canary_traffic_percentage = 0.0

            rollback_record = {
                "timestamp": time.time(),
                "version": version,
                "canary_percentage": 0.0,
                "status": "rollback",
            }
            self.deployment_history.append(rollback_record)

            return True

        except Exception as e:
            logging.error(f"Canary rollback failed: {e}")
            return False

    def health_check(self) -> bool:
        """Check canary health and potentially increase traffic."""
        if self.canary_traffic_percentage == 0.0:
            return True

        # Check canary performance
        recent_metrics = list(self.monitor.metrics_history)[-10:]
        if not recent_metrics:
            return True

        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])

        if avg_error_rate > self.canary_threshold:
            # Rollback canary
            self.canary_traffic_percentage = 0.0
            return False
        elif avg_error_rate < self.canary_threshold * 0.5:
            # Increase canary traffic
            self.canary_traffic_percentage = min(
                1.0, self.canary_traffic_percentage * 2
            )

        return True


class MicroserviceArchitecture:
    """Microservices architecture for RL systems."""

    def __init__(self):
        self.services = {}
        self.service_dependencies = {}
        self.load_balancer = None
        self.service_discovery = {}

    def add_service(self, name: str, service_type: str, config: Dict[str, Any]):
        """Add a microservice."""
        service = {
            "name": name,
            "type": service_type,
            "config": config,
            "status": "running",
            "instances": [],
            "metrics": deque(maxlen=1000),
        }
        self.services[name] = service

    def add_service_dependency(self, service_name: str, dependency_name: str):
        """Add a dependency between services."""
        if service_name not in self.service_dependencies:
            self.service_dependencies[service_name] = []
        self.service_dependencies[service_name].append(dependency_name)

    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status of a service."""
        if service_name not in self.services:
            return {"status": "not_found"}

        service = self.services[service_name]
        recent_metrics = list(service["metrics"])[-10:]

        if not recent_metrics:
            return {"status": "unknown"}

        avg_latency = np.mean([m.get("latency", 0) for m in recent_metrics])
        avg_error_rate = np.mean([m.get("error_rate", 0) for m in recent_metrics])

        health_status = "healthy"
        if avg_error_rate > 0.1:
            health_status = "unhealthy"
        elif avg_latency > 1000:
            health_status = "degraded"

        return {
            "status": health_status,
            "avg_latency": avg_latency,
            "avg_error_rate": avg_error_rate,
            "instances": len(service["instances"]),
        }

    def scale_service(self, service_name: str, target_instances: int):
        """Scale a service to target number of instances."""
        if service_name not in self.services:
            return False

        service = self.services[service_name]
        current_instances = len(service["instances"])

        if target_instances > current_instances:
            # Scale up
            for i in range(target_instances - current_instances):
                instance_id = f"{service_name}_instance_{len(service['instances'])}"
                service["instances"].append(
                    {
                        "id": instance_id,
                        "status": "running",
                        "start_time": time.time(),
                    }
                )
        elif target_instances < current_instances:
            # Scale down
            service["instances"] = service["instances"][:target_instances]

        return True


class ProductionRLSystem:
    """Production-ready RL system with monitoring and deployment capabilities."""

    def __init__(self, model: nn.Module, config: DeploymentConfig):
        self.model = model
        self.config = config
        self.monitor = PerformanceMonitor()
        self.deployment_strategy = None
        self.microservices = MicroserviceArchitecture()

        # System state
        self.is_running = False
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "peak_throughput": 0.0,
        }

        # Threading
        self.processing_thread = None
        self.monitoring_thread = None

    def set_deployment_strategy(self, strategy: ModelDeploymentStrategy):
        """Set the deployment strategy."""
        self.deployment_strategy = strategy

    def start(self):
        """Start the production system."""
        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_requests)
        self.processing_thread.start()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_system)
        self.monitoring_thread.start()

        logging.info("Production RL system started")

    def stop(self):
        """Stop the production system."""
        self.is_running = False

        if self.processing_thread:
            self.processing_thread.join()
        if self.monitoring_thread:
            self.monitoring_thread.join()

        logging.info("Production RL system stopped")

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Make a prediction using the model."""
        start_time = time.time()

        try:
            with torch.no_grad():
                output = self.model(input_data)

            prediction_time = time.time() - start_time

            # Record metrics
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=0.0,  # Would need GPU monitoring
                latency=prediction_time * 1000,  # Convert to ms
                throughput=1.0 / prediction_time,
                error_rate=0.0,
                model_accuracy=1.0,  # Would need actual accuracy measurement
                prediction_time=prediction_time,
                queue_size=self.request_queue.qsize(),
            )

            self.monitor.add_metrics(metrics)
            self.performance_stats["successful_requests"] += 1

            return output

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            self.performance_stats["failed_requests"] += 1

            # Record error metrics
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=0.0,
                latency=(time.time() - start_time) * 1000,
                throughput=0.0,
                error_rate=1.0,
                model_accuracy=0.0,
                prediction_time=time.time() - start_time,
                queue_size=self.request_queue.qsize(),
            )

            self.monitor.add_metrics(metrics)
            raise

        finally:
            self.performance_stats["total_requests"] += 1

    def _process_requests(self):
        """Process requests in a separate thread."""
        while self.is_running:
            try:
                # Get request from queue (with timeout)
                request = self.request_queue.get(timeout=1.0)

                # Process request
                input_data = request["input_data"]
                request_id = request["request_id"]

                output = self.predict(input_data)

                # Send response
                response = {
                    "request_id": request_id,
                    "output": output,
                    "timestamp": time.time(),
                }
                self.response_queue.put(response)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing request: {e}")

    def _monitor_system(self):
        """Monitor system performance in a separate thread."""
        while self.is_running:
            try:
                # Collect system metrics
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    gpu_usage=0.0,
                    latency=0.0,
                    throughput=0.0,
                    error_rate=0.0,
                    model_accuracy=1.0,
                    prediction_time=0.0,
                    queue_size=self.request_queue.qsize(),
                )

                self.monitor.add_metrics(metrics)

                # Check deployment health
                if self.deployment_strategy:
                    if not self.deployment_strategy.health_check():
                        logging.warning("Deployment health check failed")

                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logging.error(f"Error in monitoring thread: {e}")
                time.sleep(10)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "is_running": self.is_running,
            "performance_stats": self.performance_stats,
            "performance_summary": self.monitor.get_performance_summary(),
            "queue_size": self.request_queue.qsize(),
            "deployment_strategy": (
                type(self.deployment_strategy).__name__
                if self.deployment_strategy
                else None
            ),
        }
