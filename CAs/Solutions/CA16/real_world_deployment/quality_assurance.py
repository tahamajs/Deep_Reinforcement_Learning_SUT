"""
Quality Assurance and Testing Framework

This module contains classes for quality assurance, testing frameworks,
validation suites, and performance benchmarking in RL systems.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import random
import json
from pathlib import Path


@dataclass
class TestResult:
    """Result of a test case."""

    test_name: str
    test_type: str
    status: str  # "passed", "failed", "skipped"
    execution_time: float
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class ValidationResult:
    """Result of validation."""

    validation_name: str
    status: str
    score: float
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: float = 0.0


@dataclass
class BenchmarkResult:
    """Result of benchmarking."""

    benchmark_name: str
    model_name: str
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    comparison_results: Dict[str, Any]
    timestamp: float = 0.0


class TestingFramework:
    """Comprehensive testing framework for RL systems."""

    def __init__(self):
        self.test_cases = {}
        self.test_results = deque(maxlen=1000)
        self.test_suites = {}
        self.test_statistics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "avg_execution_time": 0.0,
        }

    def add_test_case(
        self, test_name: str, test_function: Callable, test_type: str = "unit"
    ):
        """Add a test case to the framework."""
        self.test_cases[test_name] = {
            "function": test_function,
            "type": test_type,
            "created_at": time.time(),
        }

    def create_test_suite(self, suite_name: str, test_names: List[str]):
        """Create a test suite with multiple test cases."""
        self.test_suites[suite_name] = {
            "test_names": test_names,
            "created_at": time.time(),
        }

    def run_test(self, test_name: str, *args, **kwargs) -> TestResult:
        """Run a single test case."""
        if test_name not in self.test_cases:
            return TestResult(
                test_name=test_name,
                test_type="unknown",
                status="failed",
                execution_time=0.0,
                metrics={},
                error_message=f"Test case '{test_name}' not found",
                timestamp=time.time(),
            )

        test_case = self.test_cases[test_name]
        start_time = time.time()

        try:
            # Run the test
            result = test_case["function"](*args, **kwargs)
            execution_time = time.time() - start_time

            # Create test result
            test_result = TestResult(
                test_name=test_name,
                test_type=test_case["type"],
                status="passed" if result else "failed",
                execution_time=execution_time,
                metrics={
                    "result": float(result) if isinstance(result, (int, float)) else 1.0
                },
                timestamp=time.time(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            test_result = TestResult(
                test_name=test_name,
                test_type=test_case["type"],
                status="failed",
                execution_time=execution_time,
                metrics={},
                error_message=str(e),
                timestamp=time.time(),
            )

        # Store result
        self.test_results.append(test_result)
        self._update_statistics(test_result)

        return test_result

    def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run a test suite."""
        if suite_name not in self.test_suites:
            logging.error(f"Test suite '{suite_name}' not found")
            return []

        suite = self.test_suites[suite_name]
        results = []

        for test_name in suite["test_names"]:
            result = self.run_test(test_name)
            results.append(result)

        return results

    def run_all_tests(self) -> List[TestResult]:
        """Run all test cases."""
        results = []
        for test_name in self.test_cases.keys():
            result = self.run_test(test_name)
            results.append(result)
        return results

    def _update_statistics(self, test_result: TestResult):
        """Update test statistics."""
        self.test_statistics["total_tests"] += 1

        if test_result.status == "passed":
            self.test_statistics["passed_tests"] += 1
        elif test_result.status == "failed":
            self.test_statistics["failed_tests"] += 1
        else:
            self.test_statistics["skipped_tests"] += 1

        # Update average execution time
        total_time = sum(r.execution_time for r in self.test_results)
        self.test_statistics["avg_execution_time"] = total_time / len(self.test_results)

    def get_test_report(self) -> Dict[str, Any]:
        """Get comprehensive test report."""
        return {
            "test_statistics": self.test_statistics,
            "recent_results": list(self.test_results)[-10:],
            "test_cases": list(self.test_cases.keys()),
            "test_suites": list(self.test_suites.keys()),
            "success_rate": self.test_statistics["passed_tests"]
            / max(1, self.test_statistics["total_tests"]),
        }


class ValidationSuite:
    """Validation suite for RL models."""

    def __init__(self, model: nn.Module, state_dim: int, action_dim: int):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.validation_results = deque(maxlen=1000)
        self.validation_metrics = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "avg_validation_score": 0.0,
        }

    def validate_model_architecture(self) -> ValidationResult:
        """Validate model architecture."""
        start_time = time.time()

        try:
            # Check model structure
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            # Check input/output dimensions
            test_input = torch.randn(1, self.state_dim)
            with torch.no_grad():
                output = self.model(test_input)

            output_dim = output.shape[-1] if output.dim() > 1 else 1

            # Calculate validation score
            score = 1.0
            if output_dim != self.action_dim:
                score -= 0.5
            if total_params == 0:
                score -= 0.3
            if trainable_params == 0:
                score -= 0.2

            status = "passed" if score >= 0.8 else "failed"

            result = ValidationResult(
                validation_name="model_architecture",
                status=status,
                score=score,
                metrics={
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "output_dimension": output_dim,
                    "execution_time": time.time() - start_time,
                },
                recommendations=self._generate_architecture_recommendations(
                    score, total_params, output_dim
                ),
                timestamp=time.time(),
            )

        except Exception as e:
            result = ValidationResult(
                validation_name="model_architecture",
                status="failed",
                score=0.0,
                metrics={"error": str(e)},
                recommendations=["Fix model architecture issues"],
                timestamp=time.time(),
            )

        self.validation_results.append(result)
        self._update_validation_metrics(result)

        return result

    def validate_training_stability(
        self, training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> ValidationResult:
        """Validate training stability."""
        start_time = time.time()

        try:
            # Test training on small batch
            test_batch = training_data[:10]
            states = torch.stack([item[0] for item in test_batch])
            actions = torch.stack([item[1] for item in test_batch])
            rewards = torch.stack([item[2] for item in test_batch])

            # Forward pass
            action_logits = self.model(states)
            action_probs = F.softmax(action_logits, dim=-1)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
            loss = -(log_probs * rewards.unsqueeze(1)).mean()

            # Check for NaN or Inf
            has_nan = torch.isnan(loss).any()
            has_inf = torch.isinf(loss).any()

            # Calculate validation score
            score = 1.0
            if has_nan:
                score -= 0.5
            if has_inf:
                score -= 0.5
            if loss.item() > 100:  # Unusually high loss
                score -= 0.2

            status = "passed" if score >= 0.8 else "failed"

            result = ValidationResult(
                validation_name="training_stability",
                status=status,
                score=score,
                metrics={
                    "loss_value": loss.item(),
                    "has_nan": has_nan.item(),
                    "has_inf": has_inf.item(),
                    "execution_time": time.time() - start_time,
                },
                recommendations=self._generate_stability_recommendations(
                    score, has_nan, has_inf
                ),
                timestamp=time.time(),
            )

        except Exception as e:
            result = ValidationResult(
                validation_name="training_stability",
                status="failed",
                score=0.0,
                metrics={"error": str(e)},
                recommendations=["Fix training stability issues"],
                timestamp=time.time(),
            )

        self.validation_results.append(result)
        self._update_validation_metrics(result)

        return result

    def validate_inference_consistency(
        self, test_states: torch.Tensor
    ) -> ValidationResult:
        """Validate inference consistency."""
        start_time = time.time()

        try:
            # Run inference multiple times
            predictions = []
            for _ in range(5):
                with torch.no_grad():
                    pred = self.model(test_states)
                    predictions.append(pred)

            # Check consistency
            pred_tensor = torch.stack(predictions)
            consistency = torch.std(pred_tensor, dim=0).mean().item()

            # Calculate validation score
            score = max(0.0, 1.0 - consistency * 10)  # Lower consistency is better

            status = "passed" if score >= 0.8 else "failed"

            result = ValidationResult(
                validation_name="inference_consistency",
                status=status,
                score=score,
                metrics={
                    "consistency_std": consistency,
                    "num_predictions": len(predictions),
                    "execution_time": time.time() - start_time,
                },
                recommendations=self._generate_consistency_recommendations(
                    score, consistency
                ),
                timestamp=time.time(),
            )

        except Exception as e:
            result = ValidationResult(
                validation_name="inference_consistency",
                status="failed",
                score=0.0,
                metrics={"error": str(e)},
                recommendations=["Fix inference consistency issues"],
                timestamp=time.time(),
            )

        self.validation_results.append(result)
        self._update_validation_metrics(result)

        return result

    def _generate_architecture_recommendations(
        self, score: float, total_params: int, output_dim: int
    ) -> List[str]:
        """Generate architecture recommendations."""
        recommendations = []

        if score < 0.8:
            recommendations.append("Review model architecture")

        if total_params == 0:
            recommendations.append("Model has no parameters")

        if output_dim != self.action_dim:
            recommendations.append(
                f"Output dimension mismatch: expected {self.action_dim}, got {output_dim}"
            )

        return recommendations

    def _generate_stability_recommendations(
        self, score: float, has_nan: bool, has_inf: bool
    ) -> List[str]:
        """Generate stability recommendations."""
        recommendations = []

        if score < 0.8:
            recommendations.append("Review training stability")

        if has_nan:
            recommendations.append("Address NaN values in training")

        if has_inf:
            recommendations.append("Address infinite values in training")

        return recommendations

    def _generate_consistency_recommendations(
        self, score: float, consistency: float
    ) -> List[str]:
        """Generate consistency recommendations."""
        recommendations = []

        if score < 0.8:
            recommendations.append("Improve inference consistency")

        if consistency > 0.1:
            recommendations.append("Model predictions are inconsistent")

        return recommendations

    def _update_validation_metrics(self, result: ValidationResult):
        """Update validation metrics."""
        self.validation_metrics["total_validations"] += 1

        if result.status == "passed":
            self.validation_metrics["passed_validations"] += 1
        else:
            self.validation_metrics["failed_validations"] += 1

        # Update average score
        total_score = sum(r.score for r in self.validation_results)
        self.validation_metrics["avg_validation_score"] = total_score / len(
            self.validation_results
        )

    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            "validation_metrics": self.validation_metrics,
            "recent_results": list(self.validation_results)[-10:],
            "success_rate": self.validation_metrics["passed_validations"]
            / max(1, self.validation_metrics["total_validations"]),
        }


class PerformanceBenchmark:
    """Performance benchmarking for RL models."""

    def __init__(self):
        self.benchmark_results = deque(maxlen=1000)
        self.benchmark_metrics = {
            "total_benchmarks": 0,
            "avg_inference_time": 0.0,
            "avg_memory_usage": 0.0,
            "avg_throughput": 0.0,
        }

    def benchmark_inference_speed(
        self, model: nn.Module, test_states: torch.Tensor, num_iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark inference speed."""
        start_time = time.time()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_states)

        # Benchmark
        inference_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                iter_start = time.time()
                _ = model(test_states)
                iter_time = time.time() - iter_start
                inference_times.append(iter_time)

        total_time = time.time() - start_time

        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        throughput = len(test_states) / avg_inference_time

        result = BenchmarkResult(
            benchmark_name="inference_speed",
            model_name=type(model).__name__,
            performance_metrics={
                "avg_inference_time": avg_inference_time,
                "std_inference_time": std_inference_time,
                "throughput": throughput,
                "total_time": total_time,
            },
            resource_usage={},
            comparison_results={},
            timestamp=time.time(),
        )

        self.benchmark_results.append(result)
        self._update_benchmark_metrics(result)

        return result

    def benchmark_memory_usage(
        self, model: nn.Module, test_states: torch.Tensor
    ) -> BenchmarkResult:
        """Benchmark memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run inference
        with torch.no_grad():
            _ = model(test_states)

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - baseline_memory

        result = BenchmarkResult(
            benchmark_name="memory_usage",
            model_name=type(model).__name__,
            performance_metrics={},
            resource_usage={
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": peak_memory,
                "memory_usage_mb": memory_usage,
            },
            comparison_results={},
            timestamp=time.time(),
        )

        self.benchmark_results.append(result)
        self._update_benchmark_metrics(result)

        return result

    def benchmark_throughput(
        self, model: nn.Module, test_states: torch.Tensor, duration_seconds: int = 10
    ) -> BenchmarkResult:
        """Benchmark throughput over time."""
        start_time = time.time()
        num_predictions = 0

        with torch.no_grad():
            while time.time() - start_time < duration_seconds:
                _ = model(test_states)
                num_predictions += len(test_states)

        total_time = time.time() - start_time
        throughput = num_predictions / total_time

        result = BenchmarkResult(
            benchmark_name="throughput",
            model_name=type(model).__name__,
            performance_metrics={
                "total_predictions": num_predictions,
                "duration_seconds": total_time,
                "throughput": throughput,
            },
            resource_usage={},
            comparison_results={},
            timestamp=time.time(),
        )

        self.benchmark_results.append(result)
        self._update_benchmark_metrics(result)

        return result

    def compare_models(
        self, models: Dict[str, nn.Module], test_states: torch.Tensor
    ) -> BenchmarkResult:
        """Compare multiple models."""
        comparison_results = {}

        for model_name, model in models.items():
            # Benchmark each model
            speed_result = self.benchmark_inference_speed(
                model, test_states, num_iterations=50
            )
            memory_result = self.benchmark_memory_usage(model, test_states)

            comparison_results[model_name] = {
                "inference_time": speed_result.performance_metrics[
                    "avg_inference_time"
                ],
                "memory_usage": memory_result.resource_usage["memory_usage_mb"],
                "throughput": speed_result.performance_metrics["throughput"],
            }

        # Find best model
        best_model = min(
            comparison_results.keys(),
            key=lambda x: comparison_results[x]["inference_time"],
        )

        result = BenchmarkResult(
            benchmark_name="model_comparison",
            model_name="multiple_models",
            performance_metrics={},
            resource_usage={},
            comparison_results={
                "model_results": comparison_results,
                "best_model": best_model,
            },
            timestamp=time.time(),
        )

        self.benchmark_results.append(result)
        self._update_benchmark_metrics(result)

        return result

    def _update_benchmark_metrics(self, result: BenchmarkResult):
        """Update benchmark metrics."""
        self.benchmark_metrics["total_benchmarks"] += 1

        # Update averages
        if "inference_time" in result.performance_metrics:
            total_time = sum(
                r.performance_metrics.get("inference_time", 0)
                for r in self.benchmark_results
            )
            self.benchmark_metrics["avg_inference_time"] = total_time / len(
                self.benchmark_results
            )

        if "memory_usage_mb" in result.resource_usage:
            total_memory = sum(
                r.resource_usage.get("memory_usage_mb", 0)
                for r in self.benchmark_results
            )
            self.benchmark_metrics["avg_memory_usage"] = total_memory / len(
                self.benchmark_results
            )

        if "throughput" in result.performance_metrics:
            total_throughput = sum(
                r.performance_metrics.get("throughput", 0)
                for r in self.benchmark_results
            )
            self.benchmark_metrics["avg_throughput"] = total_throughput / len(
                self.benchmark_results
            )

    def get_benchmark_report(self) -> Dict[str, Any]:
        """Get benchmark report."""
        return {
            "benchmark_metrics": self.benchmark_metrics,
            "recent_results": list(self.benchmark_results)[-10:],
        }


class ReliabilityTester:
    """Test system reliability and fault tolerance."""

    def __init__(self):
        self.reliability_tests = {}
        self.reliability_results = deque(maxlen=1000)
        self.reliability_metrics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "avg_reliability_score": 0.0,
        }

    def test_fault_tolerance(
        self, model: nn.Module, test_states: torch.Tensor, fault_types: List[str] = None
    ) -> Dict[str, Any]:
        """Test fault tolerance."""
        if fault_types is None:
            fault_types = ["input_corruption", "parameter_noise", "gradient_overflow"]

        results = {
            "fault_tolerance": {},
            "overall_score": 0.0,
            "recommendations": [],
        }

        for fault_type in fault_types:
            if fault_type == "input_corruption":
                score = self._test_input_corruption(model, test_states)
            elif fault_type == "parameter_noise":
                score = self._test_parameter_noise(model, test_states)
            elif fault_type == "gradient_overflow":
                score = self._test_gradient_overflow(model, test_states)
            else:
                score = 0.0

            results["fault_tolerance"][fault_type] = score

        results["overall_score"] = np.mean(list(results["fault_tolerance"].values()))

        # Generate recommendations
        if results["overall_score"] < 0.7:
            results["recommendations"].append("Improve fault tolerance")

        return results

    def _test_input_corruption(
        self, model: nn.Module, test_states: torch.Tensor
    ) -> float:
        """Test input corruption tolerance."""
        try:
            # Corrupt input
            corrupted_states = test_states + torch.randn_like(test_states) * 0.1

            # Test inference
            with torch.no_grad():
                original_output = model(test_states)
                corrupted_output = model(corrupted_states)

            # Calculate robustness
            robustness = 1.0 - F.mse_loss(original_output, corrupted_output).item()
            return max(0.0, robustness)

        except Exception:
            return 0.0

    def _test_parameter_noise(
        self, model: nn.Module, test_states: torch.Tensor
    ) -> float:
        """Test parameter noise tolerance."""
        try:
            # Add noise to parameters
            original_params = {}
            for name, param in model.named_parameters():
                original_params[name] = param.data.clone()
                param.data += torch.randn_like(param.data) * 0.01

            # Test inference
            with torch.no_grad():
                noisy_output = model(test_states)

            # Restore original parameters
            for name, param in model.named_parameters():
                param.data = original_params[name]

            with torch.no_grad():
                original_output = model(test_states)

            # Calculate robustness
            robustness = 1.0 - F.mse_loss(original_output, noisy_output).item()
            return max(0.0, robustness)

        except Exception:
            return 0.0

    def _test_gradient_overflow(
        self, model: nn.Module, test_states: torch.Tensor
    ) -> float:
        """Test gradient overflow tolerance."""
        try:
            # Test training with potential overflow
            test_states.requires_grad_(True)
            output = model(test_states)
            loss = output.sum()

            # Check for gradient overflow
            loss.backward()

            has_overflow = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_overflow = True
                        break

            return 0.0 if has_overflow else 1.0

        except Exception:
            return 0.0

    def test_robustness(
        self,
        model: nn.Module,
        test_states: torch.Tensor,
        perturbation_levels: List[float] = None,
    ) -> Dict[str, Any]:
        """Test model robustness to perturbations."""
        if perturbation_levels is None:
            perturbation_levels = [0.01, 0.05, 0.1, 0.2]

        results = {
            "perturbation_levels": perturbation_levels,
            "robustness_scores": [],
            "overall_robustness": 0.0,
        }

        for level in perturbation_levels:
            try:
                # Add perturbation
                perturbation = torch.randn_like(test_states) * level
                perturbed_states = test_states + perturbation

                # Test inference
                with torch.no_grad():
                    original_output = model(test_states)
                    perturbed_output = model(perturbed_states)

                # Calculate robustness
                robustness = 1.0 - F.mse_loss(original_output, perturbed_output).item()
                results["robustness_scores"].append(robustness)

            except Exception:
                results["robustness_scores"].append(0.0)

        results["overall_robustness"] = np.mean(results["robustness_scores"])

        return results

    def get_reliability_report(self) -> Dict[str, Any]:
        """Get reliability report."""
        return {
            "reliability_metrics": self.reliability_metrics,
            "recent_results": list(self.reliability_results)[-10:],
        }


class QualityAssurance:
    """Comprehensive quality assurance framework."""

    def __init__(self, model: nn.Module, state_dim: int, action_dim: int):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize components
        self.testing_framework = TestingFramework()
        self.validation_suite = ValidationSuite(model, state_dim, action_dim)
        self.performance_benchmark = PerformanceBenchmark()
        self.reliability_tester = ReliabilityTester()

        # QA metrics
        self.qa_metrics = {
            "total_assessments": 0,
            "passed_assessments": 0,
            "failed_assessments": 0,
            "overall_quality_score": 0.0,
        }

    def run_comprehensive_assessment(
        self, test_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive quality assessment."""
        assessment = {
            "timestamp": time.time(),
            "validation_results": {},
            "benchmark_results": {},
            "reliability_results": {},
            "overall_score": 0.0,
            "recommendations": [],
        }

        # Run validation
        validation_results = []
        validation_results.append(self.validation_suite.validate_model_architecture())
        validation_results.append(
            self.validation_suite.validate_inference_consistency(
                torch.randn(10, self.state_dim)
            )
        )

        if test_data:
            validation_results.append(
                self.validation_suite.validate_training_stability(test_data)
            )

        assessment["validation_results"] = {
            "results": validation_results,
            "success_rate": sum(1 for r in validation_results if r.status == "passed")
            / len(validation_results),
        }

        # Run benchmarks
        test_states = torch.randn(100, self.state_dim)
        benchmark_results = []
        benchmark_results.append(
            self.performance_benchmark.benchmark_inference_speed(
                self.model, test_states
            )
        )
        benchmark_results.append(
            self.performance_benchmark.benchmark_memory_usage(self.model, test_states)
        )
        benchmark_results.append(
            self.performance_benchmark.benchmark_throughput(self.model, test_states)
        )

        assessment["benchmark_results"] = {
            "results": benchmark_results,
            "avg_inference_time": np.mean(
                [
                    r.performance_metrics.get("avg_inference_time", 0)
                    for r in benchmark_results
                ]
            ),
            "avg_memory_usage": np.mean(
                [r.resource_usage.get("memory_usage_mb", 0) for r in benchmark_results]
            ),
        }

        # Run reliability tests
        reliability_results = self.reliability_tester.test_fault_tolerance(
            self.model, test_states
        )
        robustness_results = self.reliability_tester.test_robustness(
            self.model, test_states
        )

        assessment["reliability_results"] = {
            "fault_tolerance": reliability_results,
            "robustness": robustness_results,
        }

        # Calculate overall score
        validation_score = assessment["validation_results"]["success_rate"]
        benchmark_score = min(
            1.0, 1.0 - assessment["benchmark_results"]["avg_inference_time"] / 1.0
        )  # Normalize
        reliability_score = reliability_results["overall_score"]
        robustness_score = robustness_results["overall_robustness"]

        assessment["overall_score"] = np.mean(
            [validation_score, benchmark_score, reliability_score, robustness_score]
        )

        # Generate recommendations
        assessment["recommendations"] = self._generate_qa_recommendations(assessment)

        # Update metrics
        self.qa_metrics["total_assessments"] += 1
        if assessment["overall_score"] >= 0.8:
            self.qa_metrics["passed_assessments"] += 1
        else:
            self.qa_metrics["failed_assessments"] += 1

        self.qa_metrics["overall_quality_score"] = assessment["overall_score"]

        return assessment

    def _generate_qa_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate QA recommendations."""
        recommendations = []

        # Validation recommendations
        if assessment["validation_results"]["success_rate"] < 0.8:
            recommendations.append("Address validation failures")

        # Benchmark recommendations
        if assessment["benchmark_results"]["avg_inference_time"] > 0.1:
            recommendations.append("Optimize inference speed")

        if assessment["benchmark_results"]["avg_memory_usage"] > 100:
            recommendations.append("Reduce memory usage")

        # Reliability recommendations
        if assessment["reliability_results"]["fault_tolerance"]["overall_score"] < 0.7:
            recommendations.append("Improve fault tolerance")

        if assessment["reliability_results"]["robustness"]["overall_robustness"] < 0.7:
            recommendations.append("Improve robustness to perturbations")

        return recommendations

    def get_qa_report(self) -> Dict[str, Any]:
        """Get comprehensive QA report."""
        return {
            "qa_metrics": self.qa_metrics,
            "testing_report": self.testing_framework.get_test_report(),
            "validation_report": self.validation_suite.get_validation_report(),
            "benchmark_report": self.performance_benchmark.get_benchmark_report(),
            "reliability_report": self.reliability_tester.get_reliability_report(),
        }
