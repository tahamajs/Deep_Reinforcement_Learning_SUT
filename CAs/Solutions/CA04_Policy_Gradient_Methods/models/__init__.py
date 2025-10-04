"""
Model utilities for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime


class ModelManager:
    """Manage model saving, loading, and versioning"""

    def __init__(self, base_path: str = "models/saved_models"):
        """Initialize model manager

        Args:
            base_path: Base path for model storage
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.model_registry = {}

    def register_model(
        self, name: str, model: nn.Module, metadata: Dict[str, Any] = None
    ):
        """Register a model

        Args:
            name: Model name
            model: PyTorch model
            metadata: Additional metadata
        """
        self.model_registry[name] = {
            "model": model,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

    def save_model(
        self,
        name: str,
        filename: str = None,
        include_optimizer: bool = False,
        optimizer=None,
    ):
        """Save registered model

        Args:
            name: Registered model name
            filename: Output filename (optional)
            include_optimizer: Whether to save optimizer state
            optimizer: Optimizer to save
        """
        if name not in self.model_registry:
            raise ValueError(f"Model '{name}' not registered")

        model_info = self.model_registry[name]
        model = model_info["model"]

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.pth"

        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "metadata": model_info["metadata"],
            "timestamp": model_info["timestamp"],
        }

        if include_optimizer and optimizer is not None:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()

        filepath = os.path.join(self.base_path, filename)
        torch.save(save_dict, filepath)

        print(f"مدل '{name}' ذخیره شد: {filepath}")
        return filepath

    def load_model(
        self,
        filename: str,
        model_class=None,
        load_optimizer: bool = False,
        optimizer=None,
    ):
        """Load model from file

        Args:
            filename: Model filename
            model_class: Model class for instantiation
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer to load state into

        Returns:
            Loaded model
        """
        filepath = os.path.join(self.base_path, filename)
        save_dict = torch.load(filepath, map_location="cpu")

        if model_class is None:
            raise ValueError("model_class must be provided")

        # Create model instance
        init_params = save_dict["metadata"].get("init_params", {})
        model = model_class(**init_params)
        model.load_state_dict(save_dict["model_state_dict"])

        # Load optimizer state if requested
        if (
            load_optimizer
            and optimizer is not None
            and "optimizer_state_dict" in save_dict
        ):
            optimizer.load_state_dict(save_dict["optimizer_state_dict"])

        print(f"مدل بارگذاری شد: {filepath}")
        return model

    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models

        Returns:
            List of model information dictionaries
        """
        models = []
        for filename in os.listdir(self.base_path):
            if filename.endswith(".pth"):
                filepath = os.path.join(self.base_path, filename)
                try:
                    save_dict = torch.load(filepath, map_location="cpu")
                    models.append(
                        {
                            "filename": filename,
                            "model_class": save_dict.get("model_class", "Unknown"),
                            "timestamp": save_dict.get("timestamp", "Unknown"),
                            "metadata": save_dict.get("metadata", {}),
                            "file_size": os.path.getsize(filepath),
                        }
                    )
                except Exception as e:
                    print(f"خطا در خواندن مدل {filename}: {e}")

        return sorted(models, key=lambda x: x["timestamp"], reverse=True)


class ModelAnalyzer:
    """Analyze model properties and performance"""

    def __init__(self):
        """Initialize model analyzer"""
        pass

    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters

        Args:
            model: PyTorch model

        Returns:
            Parameter count dictionary
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }

    def analyze_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model size and memory usage

        Args:
            model: PyTorch model

        Returns:
            Size analysis dictionary
        """
        param_info = self.count_parameters(model)

        # Estimate memory usage (in MB)
        total_memory = total_params * 4 / (1024 * 1024)  # Assuming float32

        # Model complexity metrics
        layers = list(model.modules())
        layer_types = {}
        for layer in layers:
            layer_type = layer.__class__.__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

        return {
            "parameter_count": param_info,
            "estimated_memory_mb": total_memory,
            "layer_count": len(layers),
            "layer_types": layer_types,
            "model_depth": self._calculate_depth(model),
        }

    def _calculate_depth(self, model: nn.Module) -> int:
        """Calculate model depth

        Args:
            model: PyTorch model

        Returns:
            Model depth
        """

        def get_depth(module, current_depth=0):
            if not list(module.children()):
                return current_depth
            return max(
                get_depth(child, current_depth + 1) for child in module.children()
            )

        return get_depth(model)

    def compare_models(self, models: Dict[str, nn.Module]) -> Dict[str, Any]:
        """Compare multiple models

        Args:
            models: Dictionary of model names to models

        Returns:
            Comparison results
        """
        comparison = {}

        for name, model in models.items():
            comparison[name] = self.analyze_model_size(model)

        return comparison


class ModelValidator:
    """Validate model correctness and performance"""

    def __init__(self):
        """Initialize model validator"""
        pass

    def validate_model_output(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        expected_output_shape: Tuple[int, ...] = None,
    ) -> Dict[str, Any]:
        """Validate model output shape and properties

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            expected_output_shape: Expected output shape

        Returns:
            Validation results
        """
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)

        try:
            with torch.no_grad():
                output = model(dummy_input)

            validation_results = {
                "success": True,
                "input_shape": dummy_input.shape,
                "output_shape": output.shape,
                "output_range": (output.min().item(), output.max().item()),
                "output_mean": output.mean().item(),
                "output_std": output.std().item(),
                "has_nan": torch.isnan(output).any().item(),
                "has_inf": torch.isinf(output).any().item(),
            }

            if expected_output_shape:
                validation_results["shape_match"] = (
                    output.shape[1:] == expected_output_shape
                )

        except Exception as e:
            validation_results = {"success": False, "error": str(e)}

        return validation_results

    def validate_policy_model(
        self, model: nn.Module, state_size: int, action_size: int
    ) -> Dict[str, Any]:
        """Validate policy model specifically

        Args:
            model: Policy model
            state_size: State space dimension
            action_size: Action space dimension

        Returns:
            Validation results
        """
        results = {}

        # Test forward pass
        dummy_state = torch.randn(1, state_size)

        try:
            model.eval()
            with torch.no_grad():
                if hasattr(model, "get_action_probs"):
                    probs = model.get_action_probs(dummy_state)
                    results["action_probs_shape"] = probs.shape
                    results["probabilities_sum"] = probs.sum().item()
                    results["probabilities_valid"] = torch.all(probs >= 0).item()

                if hasattr(model, "sample_action"):
                    action, log_prob = model.sample_action(dummy_state)
                    results["action_type"] = type(action)
                    results["log_prob_shape"] = log_prob.shape
                    results["log_prob_valid"] = not torch.isnan(log_prob).item()

                if hasattr(model, "forward"):
                    output = model(dummy_state)
                    results["forward_output_shape"] = output.shape

        except Exception as e:
            results["error"] = str(e)

        return results


def create_model_summary(model: nn.Module) -> str:
    """Create human-readable model summary

    Args:
        model: PyTorch model

    Returns:
        Model summary string
    """
    analyzer = ModelAnalyzer()
    size_info = analyzer.analyze_model_size(model)

    summary = f"""
خلاصه مدل: {model.__class__.__name__}
=====================================
تعداد کل پارامترها: {size_info['parameter_count']['total_parameters']:,}
پارامترهای قابل آموزش: {size_info['parameter_count']['trainable_parameters']:,}
حافظه تخمینی: {size_info['estimated_memory_mb']:.2f} MB
تعداد لایه‌ها: {size_info['layer_count']}
عمق مدل: {size_info['model_depth']}

انواع لایه‌ها:
"""

    for layer_type, count in size_info["layer_types"].items():
        summary += f"  {layer_type}: {count}\n"

    return summary


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filename: str,
):
    """Save model checkpoint with training state

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filename: Checkpoint filename
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs("models/saved_models", exist_ok=True)
    filepath = os.path.join("models/saved_models", filename)
    torch.save(checkpoint, filepath)

    print(f"چک‌پوینت ذخیره شد: {filepath}")


def load_model_checkpoint(
    filename: str, model: nn.Module, optimizer: torch.optim.Optimizer = None
):
    """Load model checkpoint

    Args:
        filename: Checkpoint filename
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Checkpoint information
    """
    filepath = os.path.join("models/saved_models", filename)
    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"چک‌پوینت بارگذاری شد: {filepath}")
    return checkpoint

