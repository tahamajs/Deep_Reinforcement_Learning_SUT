"""
CA16: Cutting-Edge Deep Reinforcement Learning
Foundation Models, Neurosymbolic RL, and Future Paradigms

This module contains implementations of state-of-the-art RL techniques including:
- Foundation Models (Decision Transformers)
- Neurosymbolic RL
- Human-AI Collaboration
- Continual Learning
- Advanced Computational Paradigms
- Real-World Deployment Systems
"""

# Import main classes from each module
from .foundation_models import (
    DecisionTransformer,
    FoundationModelTrainer,
    ScalingAnalyzer,
)

from .neurosymbolic import (
    NeurosymbolicAgent,
    SymbolicKnowledgeBase,
)

from .human_ai_collaboration import (
    CollaborativeAgent,
)

from .continual_learning import (
    ContinualLearningAgent,
)

from .environments import (
    SymbolicGridWorld,
    CollaborativeGridWorld,
)

from .advanced_computational import (
    QuantumInspiredRL,
    NeuromorphicNetwork,
)

from .real_world_deployment import (
    ProductionRLSystem,
    SafetyMonitor,
)

__all__ = [
    # Foundation Models
    "DecisionTransformer",
    "FoundationModelTrainer",
    "ScalingAnalyzer",
    # Neurosymbolic RL
    "NeurosymbolicAgent",
    "SymbolicKnowledgeBase",
    # Human-AI Collaboration
    "CollaborativeAgent",
    # Continual Learning
    "ContinualLearningAgent",
    # Environments
    "SymbolicGridWorld",
    "CollaborativeGridWorld",
    # Advanced Computational
    "QuantumInspiredRL",
    "NeuromorphicNetwork",
    # Real-World Deployment
    "ProductionRLSystem",
    "SafetyMonitor",
]
