"""
Advanced Algorithms Package for CA8
===================================

This package contains state-of-the-art algorithms for causal reasoning and multi-modal RL:
- Advanced Causal Discovery Algorithms
- Advanced Multi-Modal Fusion Methods
- Advanced Counterfactual Reasoning
- Advanced Meta-Learning and Transfer Learning

Author: DRL Course Team
"""

from .advanced_causal_discovery import (
    AdvancedCausalDiscovery,
    run_advanced_causal_discovery_comparison,
)

from .advanced_multimodal_fusion import (
    TransformerCrossModalAttention,
    HierarchicalMultiModalFusion,
    DynamicAdaptiveFusion,
    MemoryAugmentedFusion,
    GraphNeuralNetworkFusion,
    QuantumInspiredFusion,
    NeuromorphicFusion,
    MetaLearningFusion,
    run_advanced_multimodal_fusion_comparison,
)

from .advanced_counterfactual_reasoning import (
    StructuralCausalModel,
    CounterfactualNeuralNetwork,
    InterventionEffectEstimator,
    CausalMediationAnalysis,
    RobustCausalInference,
    MultiModalCounterfactuals,
    TemporalCausalReasoning,
    CausalExplanationGenerator,
    run_advanced_counterfactual_analysis,
)

from .advanced_meta_transfer_learning import (
    MAMLCausalLearner,
    PrototypicalCausalNetworks,
    MetaTransferCausalLearner,
    FewShotCausalLearner,
    DomainAdaptationCausalRL,
    MultiTaskCausalLearner,
    ContinualCausalLearner,
    NeuralArchitectureSearchCausal,
    run_advanced_meta_transfer_learning_comparison,
)

__version__ = "1.0.0"
__all__ = [
    # Advanced Causal Discovery
    "AdvancedCausalDiscovery",
    "run_advanced_causal_discovery_comparison",
    # Advanced Multi-Modal Fusion
    "TransformerCrossModalAttention",
    "HierarchicalMultiModalFusion",
    "DynamicAdaptiveFusion",
    "MemoryAugmentedFusion",
    "GraphNeuralNetworkFusion",
    "QuantumInspiredFusion",
    "NeuromorphicFusion",
    "MetaLearningFusion",
    "run_advanced_multimodal_fusion_comparison",
    # Advanced Counterfactual Reasoning
    "StructuralCausalModel",
    "CounterfactualNeuralNetwork",
    "InterventionEffectEstimator",
    "CausalMediationAnalysis",
    "RobustCausalInference",
    "MultiModalCounterfactuals",
    "TemporalCausalReasoning",
    "CausalExplanationGenerator",
    "run_advanced_counterfactual_analysis",
    # Advanced Meta-Learning and Transfer Learning
    "MAMLCausalLearner",
    "PrototypicalCausalNetworks",
    "MetaTransferCausalLearner",
    "FewShotCausalLearner",
    "DomainAdaptationCausalRL",
    "MultiTaskCausalLearner",
    "ContinualCausalLearner",
    "NeuralArchitectureSearchCausal",
    "run_advanced_meta_transfer_learning_comparison",
]


