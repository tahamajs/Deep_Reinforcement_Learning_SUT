"""
CA16: Cutting-Edge Deep Reinforcement Learning
Foundation Models, Neurosymbolic RL, and Future Paradigms

This package contains implementations of advanced RL techniques including:
- Foundation Models in RL (Decision Transformers, Trajectory Transformers)
- Neurosymbolic Reinforcement Learning
- Human-AI Collaborative Learning
- Continual and Lifelong Learning
- Advanced Computational Paradigms (Quantum, Neuromorphic, Federated)
- Real-World Deployment Challenges

Author: CA16 Implementation
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "CA16 Implementation Team"

# Import main components
from .src.agents import *
from .src.environments import *
from .src.foundation_models import *
from .src.experiments import *
from .src.utils import *
from .src.continual_learning import *
from .src.human_ai_collaboration import *
from .src.real_world_deployment import *
from .src.deployment_ethics import *
from .src.advanced_computation import *
from .src.neurosymbolic import *

__all__ = [
    # Agents
    "DecisionTransformer",
    "NeurosymbolicAgent", 
    "CollaborativeAgent",
    "ContinualLearningAgent",
    "AdvancedComputationalAgent",
    
    # Environments
    "SymbolicGridWorld",
    "CollaborativeGridWorld",
    "ContinualLearningEnv",
    
    # Models
    "FoundationModel",
    "NeurosymbolicPolicy",
    "PreferenceRewardModel",
    
    # Utilities
    "ProductionRLSystem",
    "SafetyMonitor",
    "BiasDetector",
    "ModelVersionManager",
    "DeploymentPipeline",
]