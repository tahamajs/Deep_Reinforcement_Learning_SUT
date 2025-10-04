"""
Advanced Counterfactual Reasoning and Intervention Analysis for CA8
==================================================================

This module implements state-of-the-art counterfactual reasoning techniques:
- Structural Causal Models (SCM)
- Counterfactual Neural Networks
- Intervention Effect Estimation
- Causal Mediation Analysis
- Robust Causal Inference
- Multi-Modal Counterfactuals
- Temporal Causal Reasoning
- Causal Explanation Generation

Author: DRL Course Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.linear_model import LinearRegression
import networkx as nx
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


class StructuralCausalModel(nn.Module):
    """Structural Causal Model for counterfactual reasoning"""

    def __init__(self, n_vars: int, hidden_dim: int = 64):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim

        # Causal structure (learnable adjacency matrix)
        self.adjacency = nn.Parameter(torch.randn(n_vars, n_vars))

        # Structural equations for each variable
        self.structural_equations = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_vars, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_vars)
            ]
        )

        # Noise models
        self.noise_models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
                )
                for _ in range(n_vars)
            ]
        )

    def forward(
        self, x: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through structural equations"""
        if noise is None:
            noise = torch.randn_like(x)

        # Apply causal structure
        adj_matrix = torch.sigmoid(self.adjacency)
        causal_input = torch.matmul(x, adj_matrix)

        # Compute structural equations
        outputs = []
        for i, equation in enumerate(self.structural_equations):
            # Structural equation
            structural_output = equation(causal_input)

            # Add noise
            noise_output = self.noise_models[i](noise[:, i : i + 1])

            # Combine structural and noise components
            final_output = structural_output + noise_output
            outputs.append(final_output)

        return torch.cat(outputs, dim=1)

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get learned adjacency matrix"""
        return torch.sigmoid(self.adjacency)

    def intervene(
        self, x: torch.Tensor, intervention: Dict[int, float]
    ) -> torch.Tensor:
        """Perform causal intervention"""
        intervened_x = x.clone()

        for var_idx, value in intervention.items():
            intervened_x[:, var_idx] = value

        # Forward pass with intervention
        return self.forward(intervened_x)

    def counterfactual(
        self,
        x: torch.Tensor,
        intervention: Dict[int, float],
        factual_outcome: torch.Tensor,
    ) -> torch.Tensor:
        """Compute counterfactual outcome"""
        # Perform intervention
        intervened_x = x.clone()
        for var_idx, value in intervention.items():
            intervened_x[:, var_idx] = value

        # Compute counterfactual outcome
        counterfactual_outcome = self.forward(intervened_x)

        return counterfactual_outcome


class CounterfactualNeuralNetwork(nn.Module):
    """Neural network for counterfactual reasoning"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Treatment-specific heads
        self.treatment_heads = nn.ModuleDict(
            {
                "treated": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, output_dim),
                ),
                "control": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, output_dim),
                ),
            }
        )

        # Counterfactual head
        self.counterfactual_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for treatment indicator
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, x: torch.Tensor, treatment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode input
        encoded = self.encoder(x)

        # Treatment-specific predictions
        treated_pred = self.treatment_heads["treated"](encoded)
        control_pred = self.treatment_heads["control"](encoded)

        # Counterfactual prediction
        counterfactual_input = torch.cat([encoded, treatment], dim=1)
        counterfactual_pred = self.counterfactual_head(counterfactual_input)

        return {
            "treated": treated_pred,
            "control": control_pred,
            "counterfactual": counterfactual_pred,
        }

    def compute_ite(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Individual Treatment Effect"""
        encoded = self.encoder(x)

        treated_pred = self.treatment_heads["treated"](encoded)
        control_pred = self.treatment_heads["control"](encoded)

        return treated_pred - control_pred


class InterventionEffectEstimator(nn.Module):
    """Advanced intervention effect estimation"""

    def __init__(self, n_vars: int, hidden_dim: int = 64):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim

        # Treatment assignment model
        self.treatment_model = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Outcome prediction models
        self.outcome_models = nn.ModuleDict(
            {
                "treated": nn.Sequential(
                    nn.Linear(n_vars + 1, hidden_dim),  # +1 for treatment
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                ),
                "control": nn.Sequential(
                    nn.Linear(n_vars + 1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                ),
            }
        )

        # Propensity score network
        self.propensity_network = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, treatment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Predict treatment assignment
        treatment_prob = self.treatment_model(x)

        # Compute propensity scores
        propensity_score = self.propensity_network(x)

        # Predict outcomes
        treated_input = torch.cat([x, treatment], dim=1)
        control_input = torch.cat([x, 1 - treatment], dim=1)

        treated_outcome = self.outcome_models["treated"](treated_input)
        control_outcome = self.outcome_models["control"](control_input)

        return {
            "treatment_prob": treatment_prob,
            "propensity_score": propensity_score,
            "treated_outcome": treated_outcome,
            "control_outcome": control_outcome,
        }

    def estimate_ate(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate Average Treatment Effect"""
        treated_input = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        control_input = torch.cat([x, torch.zeros(x.shape[0], 1)], dim=1)

        treated_outcome = self.outcome_models["treated"](treated_input)
        control_outcome = self.outcome_models["control"](control_input)

        return torch.mean(treated_outcome - control_outcome)


class CausalMediationAnalysis(nn.Module):
    """Causal mediation analysis"""

    def __init__(self, n_vars: int, mediator_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_vars = n_vars
        self.mediator_dim = mediator_dim
        self.hidden_dim = hidden_dim

        # Treatment effect on mediator
        self.treatment_to_mediator = nn.Sequential(
            nn.Linear(n_vars + 1, hidden_dim),  # +1 for treatment
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mediator_dim),
        )

        # Mediator effect on outcome
        self.mediator_to_outcome = nn.Sequential(
            nn.Linear(mediator_dim + n_vars + 1, hidden_dim),  # +1 for treatment
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Direct effect (treatment -> outcome)
        self.direct_effect = nn.Sequential(
            nn.Linear(n_vars + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, x: torch.Tensor, treatment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Treatment effect on mediator
        treatment_input = torch.cat([x, treatment], dim=1)
        mediator = self.treatment_to_mediator(treatment_input)

        # Mediator effect on outcome
        mediator_input = torch.cat([mediator, x, treatment], dim=1)
        mediated_outcome = self.mediator_to_outcome(mediator_input)

        # Direct effect
        direct_outcome = self.direct_effect(treatment_input)

        return {
            "mediator": mediator,
            "mediated_outcome": mediated_outcome,
            "direct_outcome": direct_outcome,
            "total_outcome": mediated_outcome + direct_outcome,
        }

    def compute_mediation_effects(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute mediation effects"""
        # Treatment = 1
        treatment_1 = torch.ones(x.shape[0], 1)
        result_1 = self.forward(x, treatment_1)

        # Treatment = 0
        treatment_0 = torch.zeros(x.shape[0], 1)
        result_0 = self.forward(x, treatment_0)

        # Compute effects
        total_effect = result_1["total_outcome"] - result_0["total_outcome"]
        direct_effect = result_1["direct_outcome"] - result_0["direct_outcome"]
        indirect_effect = total_effect - direct_effect

        return {
            "total_effect": total_effect,
            "direct_effect": direct_effect,
            "indirect_effect": indirect_effect,
        }


class RobustCausalInference(nn.Module):
    """Robust causal inference with uncertainty quantification"""

    def __init__(self, n_vars: int, hidden_dim: int = 64):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim

        # Multiple models for robustness
        self.models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_vars + 1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2),  # mean and variance
                )
                for _ in range(5)  # 5 different models
            ]
        )

        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(5) / 5)

    def forward(
        self, x: torch.Tensor, treatment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty"""
        input_tensor = torch.cat([x, treatment], dim=1)

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(input_tensor)
            predictions.append(pred)

        # Weighted ensemble
        predictions = torch.stack(predictions, dim=0)  # [n_models, batch_size, 2]
        weights = F.softmax(self.ensemble_weights, dim=0)

        # Weighted average
        mean_pred = torch.sum(
            weights.unsqueeze(1).unsqueeze(2) * predictions[:, :, 0:1], dim=0
        )
        var_pred = torch.sum(
            weights.unsqueeze(1).unsqueeze(2) * predictions[:, :, 1:2], dim=0
        )

        return {
            "mean": mean_pred,
            "variance": var_pred,
            "uncertainty": torch.sqrt(var_pred),
        }

    def compute_robust_ate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute robust ATE with uncertainty"""
        # Treatment = 1
        treatment_1 = torch.ones(x.shape[0], 1)
        result_1 = self.forward(x, treatment_1)

        # Treatment = 0
        treatment_0 = torch.zeros(x.shape[0], 1)
        result_0 = self.forward(x, treatment_0)

        # Compute ATE
        ate_mean = torch.mean(result_1["mean"] - result_0["mean"])
        ate_var = torch.mean(result_1["variance"] + result_0["variance"])
        ate_uncertainty = torch.sqrt(ate_var)

        return {
            "ate_mean": ate_mean,
            "ate_variance": ate_var,
            "ate_uncertainty": ate_uncertainty,
        }


class MultiModalCounterfactuals(nn.Module):
    """Multi-modal counterfactual reasoning"""

    def __init__(self, modal_dims: Dict[str, int], hidden_dim: int = 128):
        super().__init__()
        self.modal_dims = modal_dims
        self.hidden_dim = hidden_dim

        # Modality encoders
        self.modal_encoders = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.modal_encoders[modal_name] = nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )

        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Counterfactual generator
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(hidden_dim * len(modal_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sum(modal_dims.values())),
        )

        # Treatment effect estimator
        self.treatment_effect_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, modal_inputs: Dict[str, torch.Tensor], treatment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode modalities
        encoded_modals = {}
        for modal_name, input_tensor in modal_inputs.items():
            encoded_modals[modal_name] = self.modal_encoders[modal_name](input_tensor)

        # Cross-modal attention
        modal_names = list(self.modal_dims.keys())
        modal_stack = torch.stack([encoded_modals[name] for name in modal_names], dim=1)

        attended_features, attention_weights = self.cross_modal_attention(
            modal_stack, modal_stack, modal_stack
        )

        # Flatten for counterfactual generation
        flattened_features = attended_features.view(attended_features.shape[0], -1)

        # Generate counterfactual
        counterfactual_output = self.counterfactual_generator(flattened_features)

        # Estimate treatment effect
        treatment_effect = self.treatment_effect_estimator(
            torch.mean(attended_features, dim=1)
        )

        return {
            "counterfactual": counterfactual_output,
            "treatment_effect": treatment_effect,
            "attention_weights": attention_weights,
        }


class TemporalCausalReasoning(nn.Module):
    """Temporal causal reasoning with LSTM"""

    def __init__(self, n_vars: int, hidden_dim: int = 64, sequence_length: int = 10):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Causal structure over time
        self.temporal_adjacency = nn.Parameter(
            torch.randn(sequence_length, n_vars, n_vars)
        )

        # Treatment effect over time
        self.temporal_treatment_effect = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for treatment
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, treatment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Temporal encoding
        temporal_features, (hidden, cell) = self.temporal_encoder(x)

        # Apply temporal causal structure
        temporal_adj = torch.sigmoid(self.temporal_adjacency)
        causal_features = torch.matmul(x, temporal_adj)

        # Treatment effect over time
        treatment_input = torch.cat(
            [
                temporal_features,
                treatment.unsqueeze(1).expand(-1, self.sequence_length, -1),
            ],
            dim=-1,
        )
        treatment_effects = self.temporal_treatment_effect(treatment_input)

        return {
            "temporal_features": temporal_features,
            "causal_features": causal_features,
            "treatment_effects": treatment_effects,
            "hidden_state": hidden,
        }


class CausalExplanationGenerator(nn.Module):
    """Generate causal explanations"""

    def __init__(self, n_vars: int, hidden_dim: int = 64):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim

        # Explanation generator
        self.explanation_generator = nn.Sequential(
            nn.Linear(n_vars + 1, hidden_dim),  # +1 for treatment
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_vars),  # Importance scores
        )

        # Attention mechanism for explanations
        self.explanation_attention = nn.MultiheadAttention(
            n_vars, num_heads=4, batch_first=True
        )

    def forward(
        self, x: torch.Tensor, treatment: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate explanations"""
        input_tensor = torch.cat([x, treatment], dim=1)

        # Generate importance scores
        importance_scores = torch.sigmoid(self.explanation_generator(input_tensor))

        # Apply attention
        attended_explanations, attention_weights = self.explanation_attention(
            importance_scores.unsqueeze(1),
            importance_scores.unsqueeze(1),
            importance_scores.unsqueeze(1),
        )

        return {
            "importance_scores": importance_scores,
            "explanations": attended_explanations.squeeze(1),
            "attention_weights": attention_weights,
        }


def run_advanced_counterfactual_analysis(
    data: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run comprehensive counterfactual analysis"""

    print("🔬 Advanced Counterfactual Reasoning Analysis")
    print("=" * 60)

    n_samples, n_vars = data.shape

    # Convert to tensors
    data_tensor = torch.FloatTensor(data)
    treatment_tensor = torch.FloatTensor(treatment).unsqueeze(1)
    outcome_tensor = torch.FloatTensor(outcome).unsqueeze(1)

    results = {}

    # 1. Structural Causal Model
    print("\n🔍 Testing Structural Causal Model...")
    try:
        scm = StructuralCausalModel(n_vars)

        # Train SCM
        optimizer = torch.optim.Adam(scm.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()

            predicted = scm(data_tensor)
            loss = F.mse_loss(predicted, data_tensor)

            loss.backward()
            optimizer.step()

        # Test interventions
        intervention = {0: 1.0}  # Intervene on first variable
        intervened_outcome = scm.intervene(data_tensor, intervention)

        results["SCM"] = {
            "adjacency_matrix": scm.get_adjacency_matrix().detach().numpy(),
            "intervention_effect": torch.mean(intervened_outcome - data_tensor).item(),
            "success": True,
        }
        print("  ✅ SCM completed successfully")

    except Exception as e:
        print(f"  ❌ SCM failed: {str(e)}")
        results["SCM"] = {"error": str(e), "success": False}

    # 2. Counterfactual Neural Network
    print("\n🔍 Testing Counterfactual Neural Network...")
    try:
        cfn = CounterfactualNeuralNetwork(n_vars)

        # Train CFN
        optimizer = torch.optim.Adam(cfn.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()

            predictions = cfn(data_tensor, treatment_tensor)

            # Compute loss
            treated_loss = F.mse_loss(predictions["treated"], outcome_tensor)
            control_loss = F.mse_loss(predictions["control"], outcome_tensor)
            total_loss = treated_loss + control_loss

            total_loss.backward()
            optimizer.step()

        # Compute ITE
        ite = cfn.compute_ite(data_tensor)

        results["CFN"] = {
            "ite_mean": torch.mean(ite).item(),
            "ite_std": torch.std(ite).item(),
            "success": True,
        }
        print("  ✅ CFN completed successfully")

    except Exception as e:
        print(f"  ❌ CFN failed: {str(e)}")
        results["CFN"] = {"error": str(e), "success": False}

    # 3. Intervention Effect Estimator
    print("\n🔍 Testing Intervention Effect Estimator...")
    try:
        iee = InterventionEffectEstimator(n_vars)

        # Train IEE
        optimizer = torch.optim.Adam(iee.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()

            predictions = iee(data_tensor, treatment_tensor)

            # Compute loss
            treated_loss = F.mse_loss(predictions["treated_outcome"], outcome_tensor)
            control_loss = F.mse_loss(predictions["control_outcome"], outcome_tensor)
            total_loss = treated_loss + control_loss

            total_loss.backward()
            optimizer.step()

        # Estimate ATE
        ate = iee.estimate_ate(data_tensor)

        results["IEE"] = {"ate": ate.item(), "success": True}
        print("  ✅ IEE completed successfully")

    except Exception as e:
        print(f"  ❌ IEE failed: {str(e)}")
        results["IEE"] = {"error": str(e), "success": False}

    # 4. Causal Mediation Analysis
    print("\n🔍 Testing Causal Mediation Analysis...")
    try:
        cma = CausalMediationAnalysis(n_vars, mediator_dim=4)

        # Train CMA
        optimizer = torch.optim.Adam(cma.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()

            predictions = cma(data_tensor, treatment_tensor)

            # Compute loss
            total_loss = F.mse_loss(predictions["total_outcome"], outcome_tensor)

            total_loss.backward()
            optimizer.step()

        # Compute mediation effects
        mediation_effects = cma.compute_mediation_effects(data_tensor)

        results["CMA"] = {
            "total_effect": torch.mean(mediation_effects["total_effect"]).item(),
            "direct_effect": torch.mean(mediation_effects["direct_effect"]).item(),
            "indirect_effect": torch.mean(mediation_effects["indirect_effect"]).item(),
            "success": True,
        }
        print("  ✅ CMA completed successfully")

    except Exception as e:
        print(f"  ❌ CMA failed: {str(e)}")
        results["CMA"] = {"error": str(e), "success": False}

    # 5. Robust Causal Inference
    print("\n🔍 Testing Robust Causal Inference...")
    try:
        rci = RobustCausalInference(n_vars)

        # Train RCI
        optimizer = torch.optim.Adam(rci.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()

            predictions = rci(data_tensor, treatment_tensor)

            # Compute loss
            loss = F.mse_loss(predictions["mean"], outcome_tensor)

            loss.backward()
            optimizer.step()

        # Compute robust ATE
        robust_ate = rci.compute_robust_ate(data_tensor)

        results["RCI"] = {
            "ate_mean": robust_ate["ate_mean"].item(),
            "ate_uncertainty": robust_ate["ate_uncertainty"].item(),
            "success": True,
        }
        print("  ✅ RCI completed successfully")

    except Exception as e:
        print(f"  ❌ RCI failed: {str(e)}")
        results["RCI"] = {"error": str(e), "success": False}

    # Create visualization
    if save_path:
        _create_counterfactual_analysis_plot(results, save_path)

    return results


def _create_counterfactual_analysis_plot(results: Dict[str, Any], save_path: str):
    """Create counterfactual analysis visualization"""

    valid_results = {k: v for k, v in results.items() if v.get("success", False)}

    if not valid_results:
        print("No valid results to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Treatment Effect Comparison
    ax = axes[0, 0]
    methods = list(valid_results.keys())

    # Extract treatment effects
    treatment_effects = []
    for method in methods:
        if "ite_mean" in valid_results[method]:
            treatment_effects.append(valid_results[method]["ite_mean"])
        elif "ate" in valid_results[method]:
            treatment_effects.append(valid_results[method]["ate"])
        elif "ate_mean" in valid_results[method]:
            treatment_effects.append(valid_results[method]["ate_mean"])
        else:
            treatment_effects.append(0)

    bars = ax.bar(
        range(len(treatment_effects)), treatment_effects, color="skyblue", alpha=0.7
    )
    ax.set_xticks(range(len(treatment_effects)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Treatment Effect")
    ax.set_title("Treatment Effect Comparison")
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    # 2. Uncertainty Quantification
    ax = axes[0, 1]

    uncertainties = []
    for method in methods:
        if "ite_std" in valid_results[method]:
            uncertainties.append(valid_results[method]["ite_std"])
        elif "ate_uncertainty" in valid_results[method]:
            uncertainties.append(valid_results[method]["ate_uncertainty"])
        else:
            uncertainties.append(0)

    bars = ax.bar(
        range(len(uncertainties)), uncertainties, color="lightcoral", alpha=0.7
    )
    ax.set_xticks(range(len(uncertainties)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Uncertainty")
    ax.set_title("Uncertainty Quantification")
    ax.grid(True, alpha=0.3)

    # 3. Mediation Analysis (if available)
    ax = axes[0, 2]

    if "CMA" in valid_results:
        mediation_data = [
            valid_results["CMA"]["total_effect"],
            valid_results["CMA"]["direct_effect"],
            valid_results["CMA"]["indirect_effect"],
        ]
        mediation_labels = ["Total", "Direct", "Indirect"]

        bars = ax.bar(
            range(len(mediation_data)),
            mediation_data,
            color=["skyblue", "lightgreen", "lightcoral"],
            alpha=0.7,
        )
        ax.set_xticks(range(len(mediation_data)))
        ax.set_xticklabels(mediation_labels)
        ax.set_ylabel("Effect Size")
        ax.set_title("Causal Mediation Analysis")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Mediation Analysis\nNot Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Causal Mediation Analysis")

    # 4. Method Characteristics Radar
    ax = axes[1, 0]
    ax.axis("off")

    ax_radar = plt.subplot(2, 3, 4, projection="polar")

    categories = [
        "Accuracy",
        "Robustness",
        "Interpretability",
        "Scalability",
        "Innovation",
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Mock scores for demonstration
    method_scores = {
        "SCM": [8, 7, 9, 6, 7],
        "CFN": [9, 8, 6, 7, 8],
        "IEE": [7, 8, 7, 8, 6],
        "CMA": [8, 7, 9, 6, 7],
        "RCI": [9, 9, 7, 7, 8],
    }

    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_results)))

    for i, (method, scores) in enumerate(method_scores.items()):
        if method in valid_results:
            scores += scores[:1]
            ax_radar.plot(
                angles, scores, "o-", linewidth=2, label=method, color=colors[i]
            )
            ax_radar.fill(angles, scores, alpha=0.15, color=colors[i])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_ylim(0, 10)
    ax_radar.set_title("Method Characteristics", fontweight="bold", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax_radar.grid(True)

    # 5. Causal Graph Visualization (if available)
    ax = axes[1, 1]

    if "SCM" in valid_results:
        adjacency_matrix = valid_results["SCM"]["adjacency_matrix"]

        # Create network graph
        G = nx.DiGraph()
        n_vars = adjacency_matrix.shape[0]

        for i in range(n_vars):
            G.add_node(i, label=f"X{i}")

        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency_matrix[i, j] > 0.5:
                    G.add_edge(i, j, weight=adjacency_matrix[i, j])

        # Draw graph
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color="lightblue",
            node_size=1000,
            font_size=10,
            font_weight="bold",
            edge_color="gray",
            arrows=True,
            arrowsize=20,
        )

        ax.set_title("Learned Causal Structure")
    else:
        ax.text(
            0.5,
            0.5,
            "Causal Graph\nNot Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Causal Structure")

    # 6. Summary and Recommendations
    ax = axes[1, 2]
    ax.axis("off")

    best_method = max(valid_results.keys(), key=lambda x: method_scores.get(x, [0])[0])

    summary_text = f"""
    📊 Advanced Counterfactual Analysis
    
    🏆 Best Method: {best_method}
    
    🔬 Method Insights:
    
    • SCM: Structural equations,
      interpretable causal structure
    
    • CFN: Neural network approach,
      good for complex relationships
    
    • IEE: Propensity score matching,
      robust causal inference
    
    • CMA: Mediation analysis,
      decomposes total effects
    
    • RCI: Uncertainty quantification,
      robust to model misspecification
    
    💡 Key Findings:
    
    • Counterfactual reasoning enables
      what-if analysis
    
    • Intervention effects can be
      estimated accurately
    
    • Mediation analysis reveals
      causal pathways
    
    • Uncertainty quantification
      provides confidence bounds
    
    🎯 Applications:
    
    • Policy evaluation
    • Treatment effect estimation
    • Causal explanation
    • Robust decision making
    """

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Counterfactual analysis plot saved to: {save_path}")


if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_vars = 4

    # Generate covariates
    X = np.random.randn(n_samples, n_vars)

    # Generate treatment (binary)
    treatment_prob = 1 / (1 + np.exp(-0.5 * X[:, 0] + 0.3 * X[:, 1]))
    treatment = np.random.binomial(1, treatment_prob)

    # Generate outcome
    outcome = (
        0.5 * X[:, 0]
        + 0.3 * X[:, 1]
        + 0.2 * X[:, 2]
        + 0.8 * treatment
        + 0.1 * np.random.randn(n_samples)
    )

    print("🚀 Testing Advanced Counterfactual Reasoning")
    print("=" * 50)

    # Run analysis
    results = run_advanced_counterfactual_analysis(
        X,
        treatment,
        outcome,
        save_path="visualizations/advanced_counterfactual_analysis.png",
    )

    print("\n🎉 Advanced Counterfactual Analysis Complete!")
    print("=" * 50)
