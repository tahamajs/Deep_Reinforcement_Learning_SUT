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
from .agents import *
from .environments import *
from .models import *
from .experiments import *
from .utils import *

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