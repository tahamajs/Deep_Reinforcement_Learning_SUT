"""
CA16: Cutting-Edge Deep Reinforcement Learning - Foundation Models, Neurosymbolic RL, and Future Paradigms

This package contains implementations of advanced deep RL paradigms including:
- Foundation Models in RL (Decision Transformers, Multi-task learning, In-context learning)
- Neurosymbolic Reinforcement Learning (Symbolic reasoning, Interpretable RL)
- Continual and Lifelong Learning (Catastrophic forgetting prevention, Meta-learning)
- Human-AI Collaborative Learning (RLHF, Preference learning, Shared autonomy)
- Advanced Computational Methods (Quantum RL, Neuromorphic computing)
- Real-World Deployment and Ethics (Production systems, Safety, Compliance)

Author: Advanced RL Research Group
Version: 1.0.0
"""

__version__ = "1.0.0"

# Import main classes for easy access
from .foundation_models.algorithms import (
    DecisionTransformer,
    MultiTaskRLFoundationModel,
    InContextLearningRL,
    FoundationModelTrainer,
)

from .neurosymbolic.knowledge_base import (
    SymbolicKnowledgeBase,
    LogicalPredicate,
    LogicalRule,
)

from .neurosymbolic.neural_components import (
    NeuralPerceptionModule,
    SymbolicReasoningModule,
)

from .neurosymbolic.policies import NeurosymbolicPolicy, NeurosymbolicAgent

from .human_ai_collaboration.collaborative_agent import CollaborativeAgent

from .human_ai_collaboration.feedback_collector import (
    HumanFeedbackCollector,
    HumanPreference,
    HumanFeedback,
)

from .human_ai_collaboration.preference_model import PreferenceRewardModel

# Continual Learning
from .continual_learning.elastic_weight_consolidation import ElasticWeightConsolidation
from .continual_learning.progressive_networks import ProgressiveNetwork
from .continual_learning.meta_learning import MAMLAgent, MetaLearner

# Advanced Computation
from .advanced_computation.quantum_rl import QuantumRLAgent
from .advanced_computation.neuromorphic_networks import NeuromorphicNetwork
from .advanced_computation.distributed_rl import DistributedRLTrainer

# Real-World Deployment
from .real_world_deployment.production_rl_agent import (
    ProductionRLAgent,
    ModelServing,
    LoadBalancer,
)
from .real_world_deployment.safety_monitor import (
    SafetyMonitor,
    RiskAssessor,
    SafetyConstraints,
)
from .real_world_deployment.ethics_checker import (
    EthicsChecker,
    BiasDetector,
    FairnessEvaluator,
)
from .real_world_deployment.deployment_framework import (
    DeploymentManager,
    MonitoringDashboard,
    RollbackSystem,
)

# Environments
from .environments.multi_modal_env import MultiModalGridWorld
from .environments.symbolic_env import SymbolicGridWorld
from .environments.collaborative_env import CollaborativeGridWorld

# Utilities
from .utils import (
    TrajectoryBuffer,
    MetricsTracker,
    ExperimentConfig,
    evaluate_policy,
    plot_training_progress,
    set_seed,
    create_mlp_network,
    compute_returns_to_go,
)

# Experiments
from .experiments import (
    FoundationModelExperiment,
    NeurosymbolicExperiment,
    ContinualLearningExperiment,
    AdvancedComputationExperiment,
    HumanAICollaborationExperiment,
    RealWorldDeploymentExperiment,
    ComprehensiveEvaluationSuite,
    run_experiment_suite,
)


# Utility functions
def create_experiment_runner():
    """Factory function to create experiment runners for different paradigms"""
    # Return a comprehensive evaluation suite instead
    from .experiments import ComprehensiveEvaluationSuite

    return ComprehensiveEvaluationSuite()


def list_available_paradigms():
    """List all available RL paradigms in this package"""
    return [
        "foundation_models",
        "neurosymbolic_rl",
        "continual_learning",
        "human_ai_collaboration",
        "advanced_computation",
        "real_world_deployment",
    ]


def get_paradigm_info(paradigm_name: str) -> dict:
    """Get information about a specific paradigm"""
    info = {
        "foundation_models": {
            "description": "Large-scale pre-trained RL models for sample-efficient learning",
            "key_components": [
                "DecisionTransformer",
                "MultiTaskRLFoundationModel",
                "InContextLearningRL",
            ],
            "applications": ["Few-shot learning", "Multi-task RL", "Generalization"],
        },
        "neurosymbolic_rl": {
            "description": "Integration of neural learning with symbolic reasoning",
            "key_components": [
                "NeurosymbolicPolicy",
                "SymbolicKnowledgeBase",
                "NeuralPerceptionModule",
            ],
            "applications": [
                "Interpretable RL",
                "Safety-critical systems",
                "Knowledge transfer",
            ],
        },
        "continual_learning": {
            "description": "Learning new tasks while retaining previous knowledge",
            "key_components": [
                "ElasticWeightConsolidation",
                "ProgressiveNetwork",
                "MetaLearner",
            ],
            "applications": ["Lifelong learning", "Adaptive systems", "Robotics"],
        },
        "human_ai_collaboration": {
            "description": "Learning from human feedback and collaborative decision-making",
            "key_components": [
                "CollaborativeAgent",
                "PreferenceRewardModel",
                "HumanFeedbackCollector",
            ],
            "applications": ["RLHF", "Human-robot interaction", "Personalized AI"],
        },
        "advanced_computation": {
            "description": "Quantum and neuromorphic computing for RL",
            "key_components": [
                "QuantumRLAgent",
                "NeuromorphicNetwork",
                "DistributedRL",
            ],
            "applications": [
                "High-performance RL",
                "Energy-efficient computing",
                "Scalable systems",
            ],
        },
        "real_world_deployment": {
            "description": "Production RL systems with safety and ethics",
            "key_components": ["ProductionRLAgent", "SafetyMonitor", "EthicsChecker"],
            "applications": ["Autonomous systems", "Healthcare", "Finance"],
        },
    }
    return info.get(paradigm_name, {"error": "Paradigm not found"})
