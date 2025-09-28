"""
Deployment Framework

This module provides deployment management, monitoring dashboards,
and rollback systems for production RL systems.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
from collections import defaultdict, deque
import threading
import time
import logging
import json
import os
from datetime import datetime, timedelta
import psutil
import GPUtil


class DeploymentManager:
    """
    Manages deployment of RL models to production.

    Handles model versioning, A/B testing, gradual rollouts, and rollback procedures.
    """

    def __init__(self, model_dir: str = "./models", config: Dict[str, Any] = None):
        self.model_dir = model_dir
        self.config = config or {}

        # Model versions
        self.current_version = None
        self.previous_versions = []
        self.version_history = []

        # Deployment state
        self.deployment_state = "idle"  # idle, deploying, deployed, rolling_back
        self.deployment_progress = 0.0

        # A/B testing
        self.ab_test_active = False
        self.ab_test_groups = {"A": [], "B": []}
        self.ab_metrics = defaultdict(dict)

        # Gradual rollout
        self.rollout_percentage = 0.0
        self.rollout_increment = 0.1

        # Monitoring
        self.performance_metrics = deque(maxlen=1000)
        self.error_logs = deque(maxlen=500)

        # Logging
        self.logger = logging.getLogger("DeploymentManager")
        self.logger.setLevel(logging.INFO)

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

    def deploy_model(
        self, model: nn.Module, version: str, metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Deploy a new model version.

        Args:
            model: PyTorch model to deploy
            version: Version identifier
            metadata: Additional deployment metadata

        Returns:
            Success status
        """
        try:
            self.logger.info(f"Starting deployment of model version {version}")

            # Save model
            model_path = os.path.join(self.model_dir, f"model_{version}.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "version": version,
                    "timestamp": time.time(),
                    "metadata": metadata or {},
                },
                model_path,
            )

            # Update version tracking
            if self.current_version:
                self.previous_versions.append(self.current_version)

            self.current_version = version
            self.version_history.append(
                {
                    "version": version,
                    "timestamp": time.time(),
                    "status": "deployed",
                    "metadata": metadata,
                }
            )

            # Start gradual rollout
            self._start_gradual_rollout()

            self.logger.info(f"Successfully deployed model version {version}")
            return True

        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            self.error_logs.append(
                {"timestamp": time.time(), "error": str(e), "operation": "deploy_model"}
            )
            return False

    def _start_gradual_rollout(self):
        """Start gradual rollout process."""
        self.deployment_state = "deploying"
        self.rollout_percentage = 0.0

        # Simulate gradual rollout (in practice, would integrate with load balancer)
        def rollout_process():
            while self.rollout_percentage < 1.0:
                time.sleep(10)  # Rollout increment interval
                self.rollout_percentage = min(
                    1.0, self.rollout_percentage + self.rollout_increment
                )
                self.logger.info(f"Rollout progress: {self.rollout_percentage:.1%}")

            self.deployment_state = "deployed"
            self.logger.info("Rollout completed successfully")

        rollout_thread = threading.Thread(target=rollout_process, daemon=True)
        rollout_thread.start()

    def rollback_model(self, target_version: Optional[str] = None) -> bool:
        """
        Rollback to a previous model version.

        Args:
            target_version: Version to rollback to (default: previous version)

        Returns:
            Success status
        """
        try:
            self.logger.info("Starting rollback process")

            if target_version is None:
                if not self.previous_versions:
                    raise ValueError("No previous version available for rollback")
                target_version = self.previous_versions[-1]

            if target_version not in [v["version"] for v in self.version_history]:
                raise ValueError(f"Version {target_version} not found in history")

            # Update deployment state
            self.deployment_state = "rolling_back"
            self.rollout_percentage = 0.0

            # Simulate rollback (in practice, would update load balancer)
            time.sleep(5)  # Simulated rollback time

            # Update version tracking
            self.previous_versions.append(self.current_version)
            self.current_version = target_version

            self.version_history.append(
                {
                    "version": target_version,
                    "timestamp": time.time(),
                    "status": "rolled_back",
                    "metadata": {"rollback_from": self.previous_versions[-1]},
                }
            )

            self.deployment_state = "deployed"
            self.logger.info(f"Successfully rolled back to version {target_version}")
            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            self.error_logs.append(
                {
                    "timestamp": time.time(),
                    "error": str(e),
                    "operation": "rollback_model",
                }
            )
            return False

    def start_ab_test(
        self, model_a: nn.Module, model_b: nn.Module, test_duration: int = 3600
    ) -> bool:
        """
        Start A/B testing between two models.

        Args:
            model_a: First model for testing
            model_b: Second model for testing
            test_duration: Test duration in seconds

        Returns:
            Success status
        """
        try:
            self.logger.info("Starting A/B test")

            # Deploy both models
            version_a = f"ab_test_A_{int(time.time())}"
            version_b = f"ab_test_B_{int(time.time())}"

            self.deploy_model(model_a, version_a, {"ab_test": True, "group": "A"})
            self.deploy_model(model_b, version_b, {"ab_test": True, "group": "B"})

            self.ab_test_active = True
            self.ab_test_groups = {"A": [version_a], "B": [version_b]}

            # Schedule test end
            def end_ab_test():
                time.sleep(test_duration)
                self.end_ab_test()

            test_thread = threading.Thread(target=end_ab_test, daemon=True)
            test_thread.start()

            self.logger.info("A/B test started successfully")
            return True

        except Exception as e:
            self.logger.error(f"A/B test setup failed: {str(e)}")
            return False

    def end_ab_test(self) -> Dict[str, Any]:
        """End A/B testing and return results."""
        if not self.ab_test_active:
            return {"error": "No active A/B test"}

        self.logger.info("Ending A/B test")

        # Analyze results
        results = self._analyze_ab_test_results()

        # Deploy winner or maintain current
        if results["winner"] == "A":
            self.logger.info("Model A performed better, keeping current deployment")
        elif results["winner"] == "B":
            # Deploy B as new version
            self.logger.info("Model B performed better, deploying as new version")
        else:
            self.logger.info("No clear winner, maintaining current deployment")

        self.ab_test_active = False
        return results

    def _analyze_ab_test_results(self) -> Dict[str, Any]:
        """Analyze A/B test results."""
        # Simplified analysis (would be more sophisticated in practice)
        metrics_a = self.ab_metrics.get("A", {})
        metrics_b = self.ab_metrics.get("B", {})

        # Compare key metrics (reward, latency, etc.)
        score_a = np.mean(
            [v for v in metrics_a.values() if isinstance(v, (int, float))]
        )
        score_b = np.mean(
            [v for v in metrics_b.values() if isinstance(v, (int, float))]
        )

        if score_a > score_b:
            winner = "A"
        elif score_b > score_a:
            winner = "B"
        else:
            winner = "tie"

        return {
            "winner": winner,
            "metrics_A": metrics_a,
            "metrics_B": metrics_b,
            "score_A": score_a,
            "score_B": score_b,
        }

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "current_version": self.current_version,
            "deployment_state": self.deployment_state,
            "rollout_percentage": self.rollout_percentage,
            "ab_test_active": self.ab_test_active,
            "available_versions": [v["version"] for v in self.version_history],
            "error_count": len(self.error_logs),
        }


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for deployed RL systems.

    Tracks performance metrics, system health, and alerts.
    """

    def __init__(self, update_interval: int = 60):
        self.update_interval = update_interval

        # Metrics storage
        self.metrics_history = defaultdict(deque)
        self.alerts = deque(maxlen=100)

        # Thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "gpu_usage": 90.0,
            "latency": 1000.0,  # ms
            "error_rate": 0.05,
            "reward": -float("inf"),  # Minimum acceptable reward
        }

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None

        # Logging
        self.logger = logging.getLogger("MonitoringDashboard")
        self.logger.setLevel(logging.INFO)

    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Monitoring dashboard started")

    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("Monitoring dashboard stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()

                # Store metrics
                timestamp = time.time()
                for metric, value in system_metrics.items():
                    self.metrics_history[metric].append((timestamp, value))

                # Check thresholds and generate alerts
                self._check_thresholds(system_metrics)

                # Clean old data (keep last 24 hours)
                cutoff_time = timestamp - 86400
                for metric_queue in self.metrics_history.values():
                    while metric_queue and metric_queue[0][0] < cutoff_time:
                        metric_queue.popleft()

            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")

            time.sleep(self.update_interval)

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        metrics = {}

        # CPU usage
        metrics["cpu_usage"] = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        metrics["memory_usage"] = memory.percent
        metrics["memory_used_gb"] = memory.used / (1024**3)

        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                metrics["gpu_usage"] = gpus[0].load * 100
                metrics["gpu_memory_usage"] = gpus[0].memoryUtil * 100
        except:
            metrics["gpu_usage"] = 0.0
            metrics["gpu_memory_usage"] = 0.0

        # Disk usage
        disk = psutil.disk_usage("/")
        metrics["disk_usage"] = disk.percent

        # Network I/O
        net = psutil.net_io_counters()
        metrics["network_bytes_sent"] = net.bytes_sent
        metrics["network_bytes_recv"] = net.bytes_recv

        return metrics

    def _check_thresholds(self, metrics: Dict[str, float]):
        """Check metrics against thresholds and generate alerts."""
        for metric, value in metrics.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                if metric == "reward":
                    # For reward, check if below minimum
                    if value < threshold:
                        self._generate_alert(
                            f"Low reward: {value:.3f} < {threshold}", "warning"
                        )
                else:
                    # For other metrics, check if above threshold
                    if value > threshold:
                        severity = "critical" if value > threshold * 1.2 else "warning"
                        self._generate_alert(
                            f"High {metric}: {value:.1f}% > {threshold}%", severity
                        )

    def _generate_alert(self, message: str, severity: str):
        """Generate an alert."""
        alert = {"timestamp": time.time(), "message": message, "severity": severity}

        self.alerts.append(alert)
        self.logger.warning(f"Alert generated: {message}")

    def record_performance_metric(
        self, metric_name: str, value: float, metadata: Dict[str, Any] = None
    ):
        """
        Record a custom performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        timestamp = time.time()
        self.metrics_history[metric_name].append((timestamp, value))

        # Store metadata if provided
        if metadata:
            metadata_key = f"{metric_name}_metadata"
            if metadata_key not in self.metrics_history:
                self.metrics_history[metadata_key] = deque(maxlen=1000)
            self.metrics_history[metadata_key].append((timestamp, metadata))

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for display."""
        # Get recent metrics (last hour)
        cutoff_time = time.time() - 3600

        recent_metrics = {}
        for metric, values in self.metrics_history.items():
            recent_values = [(t, v) for t, v in values if t > cutoff_time]
            if recent_values:
                recent_metrics[metric] = recent_values

        # Get recent alerts (last 24 hours)
        recent_alerts = [
            alert for alert in self.alerts if alert["timestamp"] > time.time() - 86400
        ]

        return {
            "timestamp": time.time(),
            "recent_metrics": recent_metrics,
            "recent_alerts": recent_alerts,
            "thresholds": self.thresholds,
            "system_status": self._get_system_status(),
        }

    def _get_system_status(self) -> str:
        """Get overall system status."""
        recent_alerts = [
            alert for alert in self.alerts if alert["timestamp"] > time.time() - 3600
        ]

        critical_alerts = [a for a in recent_alerts if a["severity"] == "critical"]

        if critical_alerts:
            return "critical"
        elif recent_alerts:
            return "warning"
        else:
            return "healthy"

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update monitoring thresholds."""
        self.thresholds.update(new_thresholds)
        self.logger.info(f"Updated thresholds: {new_thresholds}")


class RollbackSystem:
    """
    Automated rollback system for production RL systems.

    Monitors system health and automatically rolls back if issues are detected.
    """

    def __init__(
        self,
        deployment_manager: DeploymentManager,
        monitoring_dashboard: MonitoringDashboard,
        rollback_thresholds: Dict[str, Any] = None,
    ):
        self.deployment_manager = deployment_manager
        self.monitoring = monitoring_dashboard

        # Rollback thresholds
        self.rollback_thresholds = rollback_thresholds or {
            "error_rate_threshold": 0.1,
            "performance_drop_threshold": 0.2,
            "consecutive_failures": 5,
            "max_rollback_attempts": 3,
        }

        # Rollback state
        self.rollback_attempts = 0
        self.last_rollback_time = 0
        self.rollback_cooldown = 300  # 5 minutes

        # Monitoring state
        self.failure_count = 0
        self.baseline_performance = {}

        # Automated rollback
        self.auto_rollback_enabled = True
        self.rollback_thread = None
        self.monitoring_active = False

        # Logging
        self.logger = logging.getLogger("RollbackSystem")
        self.logger.setLevel(logging.INFO)

    def start_automated_rollback(self):
        """Start automated rollback monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.rollback_thread = threading.Thread(
            target=self._rollback_monitoring_loop, daemon=True
        )
        self.rollback_thread.start()

        # Establish baseline performance
        self._establish_baseline()

        self.logger.info("Automated rollback system started")

    def stop_automated_rollback(self):
        """Stop automated rollback monitoring."""
        self.monitoring_active = False
        if self.rollback_thread:
            self.rollback_thread.join(timeout=5)

        self.logger.info("Automated rollback system stopped")

    def _establish_baseline(self):
        """Establish baseline performance metrics."""
        self.logger.info("Establishing performance baseline...")

        # Wait for some metrics to accumulate
        time.sleep(60)

        # Get recent metrics for baseline
        dashboard_data = self.monitoring.get_dashboard_data()
        recent_metrics = dashboard_data.get("recent_metrics", {})

        for metric, values in recent_metrics.items():
            if values:
                # Use mean of recent values as baseline
                recent_values = [v for t, v in values[-10:]]  # Last 10 values
                self.baseline_performance[metric] = np.mean(recent_values)

        self.logger.info(f"Baseline established: {self.baseline_performance}")

    def _rollback_monitoring_loop(self):
        """Main rollback monitoring loop."""
        while self.monitoring_active:
            try:
                # Check if rollback conditions are met
                if self._should_rollback():
                    self._perform_automated_rollback()

            except Exception as e:
                self.logger.error(f"Rollback monitoring error: {str(e)}")

            time.sleep(30)  # Check every 30 seconds

    def _should_rollback(self) -> bool:
        """Determine if rollback conditions are met."""
        if not self.auto_rollback_enabled:
            return False

        # Check cooldown period
        if time.time() - self.last_rollback_time < self.rollback_cooldown:
            return False

        # Check maximum rollback attempts
        if self.rollback_attempts >= self.rollback_thresholds["max_rollback_attempts"]:
            self.logger.warning("Maximum rollback attempts reached")
            return False

        # Get current metrics
        dashboard_data = self.monitoring.get_dashboard_data()
        recent_metrics = dashboard_data.get("recent_metrics", {})

        # Check error rate
        if "error_rate" in recent_metrics:
            error_values = [
                v for t, v in recent_metrics["error_rate"][-5:]
            ]  # Last 5 values
            current_error_rate = np.mean(error_values)

            if current_error_rate > self.rollback_thresholds["error_rate_threshold"]:
                self.logger.warning(
                    f"High error rate detected: {current_error_rate:.3f}"
                )
                return True

        # Check performance drop
        performance_drop = self._check_performance_drop(recent_metrics)
        if performance_drop:
            self.logger.warning(f"Performance drop detected: {performance_drop}")
            return True

        # Check consecutive failures
        recent_alerts = dashboard_data.get("recent_alerts", [])
        critical_alerts = [
            a
            for a in recent_alerts
            if a["severity"] == "critical" and a["timestamp"] > time.time() - 300
        ]  # Last 5 minutes

        if len(critical_alerts) >= self.rollback_thresholds["consecutive_failures"]:
            self.logger.warning(f"Consecutive critical alerts: {len(critical_alerts)}")
            return True

        return False

    def _check_performance_drop(self, recent_metrics: Dict) -> Optional[str]:
        """Check for significant performance drops."""
        for metric, baseline in self.baseline_performance.items():
            if metric in recent_metrics:
                recent_values = [v for t, v in recent_metrics[metric][-5:]]
                if recent_values:
                    current_avg = np.mean(recent_values)
                    drop_percentage = (baseline - current_avg) / baseline

                    if (
                        drop_percentage
                        > self.rollback_thresholds["performance_drop_threshold"]
                    ):
                        return f"{metric}: {drop_percentage:.1%} drop"

        return None

    def _perform_automated_rollback(self):
        """Perform automated rollback."""
        self.logger.info("Performing automated rollback...")

        success = self.deployment_manager.rollback_model()

        if success:
            self.rollback_attempts += 1
            self.last_rollback_time = time.time()
            self.failure_count = 0  # Reset failure count

            # Re-establish baseline after rollback
            time.sleep(60)  # Wait for system to stabilize
            self._establish_baseline()

            self.logger.info("Automated rollback completed successfully")
        else:
            self.logger.error("Automated rollback failed")
            self.failure_count += 1

    def manual_rollback(self, reason: str) -> bool:
        """Perform manual rollback."""
        self.logger.info(f"Manual rollback requested: {reason}")

        success = self.deployment_manager.rollback_model()

        if success:
            self.rollback_attempts += 1
            self.last_rollback_time = time.time()

        return success

    def get_rollback_status(self) -> Dict[str, Any]:
        """Get rollback system status."""
        return {
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "rollback_attempts": self.rollback_attempts,
            "last_rollback_time": self.last_rollback_time,
            "failure_count": self.failure_count,
            "baseline_performance": self.baseline_performance,
            "thresholds": self.rollback_thresholds,
        }
