"""
CA16 Agents Package

This package contains implementations of cutting-edge RL agents including:
- Foundation model agents (Decision Transformers, Trajectory Transformers)
- Neurosymbolic agents with logic-guided policies
- Human-AI collaborative agents with preference learning
- Continual learning agents with memory and meta-learning
- Advanced computational paradigm agents (quantum, neuromorphic, federated)
"""

from .foundation_agents import (
    DecisionTransformer,
    TrajectoryTransformer,
    MultiTaskRLFoundationModel,
    InContextLearningRL,
    FoundationModelTrainer,
)

from .neurosymbolic_agents import (
    NeurosymbolicPolicy,
    NeurosymbolicAgent,
    NeuralPerceptionModule,
    SymbolicReasoningModule,
    LogicalPredicate,
    LogicalRule,
    SymbolicKnowledgeBase,
)

from .collaborative_agents import (
    CollaborativeAgent,
    PreferenceRewardModel,
    HumanFeedbackCollector,
    HumanPreference,
    HumanFeedback,
    RLHFAgent,
)

from .continual_agents import (
    ContinualLearningAgent,
    ElasticWeightConsolidation,
    ExperienceReplay,
    ProgressiveNetwork,
    MetaLearningAgent,
)

from .advanced_agents import (
    AdvancedComputationalAgent,
    QuantumInspiredRL,
    NeuromorphicNetwork,
    FederatedRLAggregator,
    EnergyEfficientRL,
)

__all__ = [
    # Foundation Agents
    "DecisionTransformer",
    "TrajectoryTransformer", 
    "MultiTaskRLFoundationModel",
    "InContextLearningRL",
    "FoundationModelTrainer",
    
    # Neurosymbolic Agents
    "NeurosymbolicPolicy",
    "NeurosymbolicAgent",
    "NeuralPerceptionModule",
    "SymbolicReasoningModule",
    "LogicalPredicate",
    "LogicalRule",
    "SymbolicKnowledgeBase",
    
    # Collaborative Agents
    "CollaborativeAgent",
    "PreferenceRewardModel",
    "HumanFeedbackCollector",
    "HumanPreference",
    "HumanFeedback",
    "RLHFAgent",
    
    # Continual Agents
    "ContinualLearningAgent",
    "ElasticWeightConsolidation",
    "ExperienceReplay",
    "ProgressiveNetwork",
    "MetaLearningAgent",
    
    # Advanced Agents
    "AdvancedComputationalAgent",
    "QuantumInspiredRL",
    "NeuromorphicNetwork",
    "FederatedRLAggregator",
    "EnergyEfficientRL",
]
