"""
Hybrid Computing Paradigms for Deep Reinforcement Learning

This module implements hybrid computing approaches combining different computational paradigms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from .quantum_rl import QuantumInspiredRL
from .neuromorphic import NeuromorphicNetwork
from .energy_efficient import EnergyEfficientRL


class HybridComputingAgent(nn.Module):
    """Base class for hybrid computing agents."""

    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Hybrid components
        self.computational_paradigms = {}
        self.paradigm_weights = {}
        self.paradigm_selector = None

        # Performance tracking
        self.performance_history = {
            "paradigm_usage": {},
            "hybrid_performance": [],
            "computational_efficiency": [],
        }

    def add_paradigm(self, name: str, paradigm: nn.Module, weight: float = 1.0):
        """Add a computational paradigm."""
        self.computational_paradigms[name] = paradigm
        self.paradigm_weights[name] = weight
        self.performance_history["paradigm_usage"][name] = []

    def select_paradigm(self, state: torch.Tensor) -> str:
        """Select the best paradigm for the current state."""
        if self.paradigm_selector is None:
            # Simple selection based on state characteristics
            state_norm = torch.norm(state).item()
            if state_norm > 1.0:
                return "quantum"
            elif state_norm > 0.5:
                return "neuromorphic"
            else:
                return "classical"
        else:
            # Use learned selector
            paradigm_scores = self.paradigm_selector(state.unsqueeze(0))
            paradigm_idx = torch.argmax(paradigm_scores, dim=-1)
            paradigm_names = list(self.computational_paradigms.keys())
            return paradigm_names[paradigm_idx.item()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid system."""
        # Select paradigm
        paradigm_name = self.select_paradigm(x)

        # Forward through selected paradigm
        paradigm = self.computational_paradigms[paradigm_name]
        output = paradigm(x)

        # Update usage statistics
        self.performance_history["paradigm_usage"][paradigm_name].append(1)

        return output

    def get_hybrid_statistics(self) -> Dict[str, Any]:
        """Get hybrid computing statistics."""
        stats = {
            "paradigm_weights": self.paradigm_weights.copy(),
            "performance_history": self.performance_history.copy(),
        }

        # Compute paradigm usage rates
        total_usage = sum(
            len(usage) for usage in self.performance_history["paradigm_usage"].values()
        )
        usage_rates = {}
        for name, usage in self.performance_history["paradigm_usage"].items():
            usage_rates[name] = len(usage) / max(total_usage, 1)

        stats["paradigm_usage_rates"] = usage_rates

        return stats


class QuantumClassicalHybrid(HybridComputingAgent):
    """Hybrid agent combining quantum and classical computing."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        quantum_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        # Quantum component
        self.quantum_component = QuantumInspiredRL(state_dim, action_dim, quantum_dim)

        # Classical component
        self.classical_component = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Hybrid fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(action_dim * 2, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

        # Paradigm selector
        self.paradigm_selector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 paradigms: quantum and classical
            nn.Softmax(dim=-1),
        )

        # Add paradigms
        self.add_paradigm("quantum", self.quantum_component, 1.0)
        self.add_paradigm("classical", self.classical_component, 1.0)

        # Hybrid parameters
        self.fusion_weight = 0.5
        self.quantum_threshold = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum-classical hybrid."""
        # Get paradigm selection
        paradigm_scores = self.paradigm_selector(x.unsqueeze(0))
        quantum_score = paradigm_scores[0, 0].item()
        classical_score = paradigm_scores[0, 1].item()

        # Get outputs from both paradigms
        quantum_output = self.quantum_component(x.unsqueeze(0))[
            0
        ]  # Get action probabilities
        classical_output = self.classical_component(x.unsqueeze(0))

        # Hybrid fusion
        if quantum_score > self.quantum_threshold:
            # Use quantum output
            output = quantum_output
            paradigm_used = "quantum"
        else:
            # Use classical output
            output = classical_output
            paradigm_used = "classical"

        # Update usage statistics
        self.performance_history["paradigm_usage"][paradigm_used].append(1)

        return output

    def get_quantum_classical_statistics(self) -> Dict[str, Any]:
        """Get quantum-classical hybrid statistics."""
        base_stats = self.get_hybrid_statistics()

        # Add quantum-specific statistics
        quantum_stats = self.quantum_component.get_quantum_statistics()
        base_stats["quantum_statistics"] = quantum_stats

        return base_stats


class NeuromorphicClassicalHybrid(HybridComputingAgent):
    """Hybrid agent combining neuromorphic and classical computing."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        tau: float = 20.0,
        threshold: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        # Neuromorphic component
        self.neuromorphic_component = NeuromorphicNetwork(
            state_dim, action_dim, hidden_dims, tau, threshold
        )

        # Classical component
        self.classical_component = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Paradigm selector
        self.paradigm_selector = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 paradigms: neuromorphic and classical
            nn.Softmax(dim=-1),
        )

        # Add paradigms
        self.add_paradigm("neuromorphic", self.neuromorphic_component, 1.0)
        self.add_paradigm("classical", self.classical_component, 1.0)

        # Hybrid parameters
        self.neuromorphic_threshold = 0.6
        self.energy_efficiency_weight = 0.3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through neuromorphic-classical hybrid."""
        # Get paradigm selection
        paradigm_scores = self.paradigm_selector(x.unsqueeze(0))
        neuromorphic_score = paradigm_scores[0, 0].item()
        classical_score = paradigm_scores[0, 1].item()

        # Get outputs from both paradigms
        neuromorphic_output = self.neuromorphic_component(x.unsqueeze(0))
        classical_output = self.classical_component(x.unsqueeze(0))

        # Hybrid selection based on energy efficiency
        neuromorphic_energy = self.neuromorphic_component.compute_energy_consumption()
        classical_energy = 1.0  # Normalized classical energy

        # Energy-efficient selection
        if (
            neuromorphic_score > self.neuromorphic_threshold
            and neuromorphic_energy < classical_energy
        ):
            output = neuromorphic_output
            paradigm_used = "neuromorphic"
        else:
            output = classical_output
            paradigm_used = "classical"

        # Update usage statistics
        self.performance_history["paradigm_usage"][paradigm_used].append(1)

        return output

    def get_neuromorphic_classical_statistics(self) -> Dict[str, Any]:
        """Get neuromorphic-classical hybrid statistics."""
        base_stats = self.get_hybrid_statistics()

        # Add neuromorphic-specific statistics
        neuromorphic_stats = self.neuromorphic_component.get_neuromorphic_statistics()
        base_stats["neuromorphic_statistics"] = neuromorphic_stats

        return base_stats


class MultiParadigmHybrid(HybridComputingAgent):
    """Hybrid agent combining multiple computational paradigms."""

    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        super().__init__(state_dim, action_dim, device)

        # Quantum component
        self.quantum_component = QuantumInspiredRL(state_dim, action_dim)

        # Neuromorphic component
        self.neuromorphic_component = NeuromorphicNetwork(state_dim, action_dim)

        # Energy-efficient component
        self.energy_efficient_component = EnergyEfficientRL(
            state_dim, action_dim, "early_exit"
        )

        # Classical component
        self.classical_component = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Multi-paradigm selector
        self.paradigm_selector = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 paradigms
            nn.Softmax(dim=-1),
        )

        # Add all paradigms
        self.add_paradigm("quantum", self.quantum_component, 1.0)
        self.add_paradigm("neuromorphic", self.neuromorphic_component, 1.0)
        self.add_paradigm("energy_efficient", self.energy_efficient_component, 1.0)
        self.add_paradigm("classical", self.classical_component, 1.0)

        # Hybrid fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(action_dim * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Performance metrics
        self.performance_metrics = {
            "accuracy": 0.0,
            "energy_efficiency": 0.0,
            "computational_speed": 0.0,
            "robustness": 0.0,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-paradigm hybrid."""
        # Get paradigm selection
        paradigm_scores = self.paradigm_selector(x.unsqueeze(0))

        # Get outputs from all paradigms
        quantum_output = self.quantum_component(x.unsqueeze(0))[0]
        neuromorphic_output = self.neuromorphic_component(x.unsqueeze(0))
        energy_efficient_output = self.energy_efficient_component(x.unsqueeze(0))
        classical_output = self.classical_component(x.unsqueeze(0))

        # Combine outputs
        combined_output = torch.cat(
            [
                quantum_output,
                neuromorphic_output,
                energy_efficient_output,
                classical_output,
            ],
            dim=-1,
        )

        # Fusion
        output = self.fusion_layer(combined_output)

        # Update usage statistics
        paradigm_names = list(self.computational_paradigms.keys())
        for i, name in enumerate(paradigm_names):
            if paradigm_scores[0, i].item() > 0.25:  # Threshold for usage
                self.performance_history["paradigm_usage"][name].append(1)

        return output

    def update_performance_metrics(
        self,
        accuracy: float,
        energy_efficiency: float,
        computational_speed: float,
        robustness: float,
    ):
        """Update performance metrics."""
        self.performance_metrics["accuracy"] = accuracy
        self.performance_metrics["energy_efficiency"] = energy_efficiency
        self.performance_metrics["computational_speed"] = computational_speed
        self.performance_metrics["robustness"] = robustness

    def get_multi_paradigm_statistics(self) -> Dict[str, Any]:
        """Get multi-paradigm hybrid statistics."""
        base_stats = self.get_hybrid_statistics()

        # Add paradigm-specific statistics
        base_stats["quantum_statistics"] = (
            self.quantum_component.get_quantum_statistics()
        )
        base_stats["neuromorphic_statistics"] = (
            self.neuromorphic_component.get_neuromorphic_statistics()
        )
        base_stats["energy_efficient_statistics"] = (
            self.energy_efficient_component.get_energy_statistics()
        )

        # Add performance metrics
        base_stats["performance_metrics"] = self.performance_metrics.copy()

        return base_stats

    def select_optimal_paradigm(
        self, state: torch.Tensor, performance_requirements: Dict[str, float]
    ) -> str:
        """Select optimal paradigm based on performance requirements."""
        paradigm_scores = self.paradigm_selector(state.unsqueeze(0))
        paradigm_names = list(self.computational_paradigms.keys())

        # Score each paradigm based on requirements
        paradigm_scores_weighted = []
        for i, name in enumerate(paradigm_names):
            score = paradigm_scores[0, i].item()

            # Weight by performance requirements
            if "accuracy" in performance_requirements:
                if name == "quantum":
                    score *= performance_requirements["accuracy"]
                elif name == "classical":
                    score *= performance_requirements["accuracy"] * 0.8

            if "energy_efficiency" in performance_requirements:
                if name == "neuromorphic":
                    score *= performance_requirements["energy_efficiency"]
                elif name == "energy_efficient":
                    score *= performance_requirements["energy_efficiency"] * 0.9

            paradigm_scores_weighted.append(score)

        # Select best paradigm
        best_idx = np.argmax(paradigm_scores_weighted)
        return paradigm_names[best_idx]
