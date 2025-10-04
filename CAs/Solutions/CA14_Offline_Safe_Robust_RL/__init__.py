"""
CA14 Advanced Deep Reinforcement Learning Package
بسته پیشرفته یادگیری تقویتی عمیق CA14

This package contains comprehensive implementations of advanced RL methods including:
- Offline Reinforcement Learning (CQL, IQL)
- Safe Reinforcement Learning (CPO, Lagrangian)
- Multi-Agent Reinforcement Learning (MADDPG, QMIX)
- Robust Reinforcement Learning (Domain Randomization, Adversarial Training)
- Advanced Algorithms (Hierarchical RL, Meta-Learning, Causal RL, Quantum RL, Neuro-Symbolic RL, Federated RL)
- Complex Environments (Multi-Objective, Partially Observable, Continuous Control, Adversarial)
- Advanced Visualizations (3D, Real-time, Multi-dimensional, Causal, Quantum, Federated)
- Advanced Concepts (Transfer Learning, Curriculum Learning, Multi-Task, Continual Learning, Explainable AI)
"""

# Version information
__version__ = "2.0.0"
__author__ = "CA14 Advanced RL Team"
__email__ = "ca14@advanced-rl.com"

# Import core modules
from . import offline_rl
from . import safe_rl
from . import multi_agent
from . import robust_rl
from . import environments
from . import evaluation
from . import utils

# Import advanced modules
from . import advanced_algorithms
from . import complex_environments
from . import advanced_visualizations
from . import advanced_concepts

# Core exports
from .offline_rl import (
    ConservativeQLearning,
    ImplicitQLearning,
    OfflineDataset,
    generate_offline_dataset
)

from .safe_rl import (
    SafeEnvironment,
    ConstrainedPolicyOptimization,
    LagrangianSafeRL,
    collect_safe_trajectory
)

from .multi_agent import (
    MultiAgentEnvironment,
    MADDPGAgent,
    QMIXAgent,
    MultiAgentReplayBuffer
)

from .robust_rl import (
    RobustEnvironment,
    DomainRandomizationAgent,
    AdversarialRobustAgent
)

from .evaluation import (
    ComprehensiveEvaluator
)

from .utils import (
    create_evaluation_environments,
    run_comprehensive_evaluation
)

# Advanced exports
from .advanced_algorithms import (
    HierarchicalRLAgent,
    MetaLearningAgent,
    CausalRLAgent,
    QuantumInspiredRLAgent,
    NeurosymbolicRLAgent,
    FederatedRLAgent
)

from .complex_environments import (
    DynamicMultiObjectiveEnvironment,
    PartiallyObservableEnvironment,
    ContinuousControlEnvironment,
    AdversarialEnvironment,
    EnvironmentConfig
)

from .advanced_visualizations import (
    Interactive3DVisualizer,
    RealTimePerformanceMonitor,
    MultiDimensionalAnalyzer,
    CausalGraphVisualizer,
    QuantumStateVisualizer,
    FederatedLearningDashboard,
    AdvancedMetricsAnalyzer,
    VisualizationConfig
)

from .advanced_concepts import (
    TransferLearningAgent,
    CurriculumLearningAgent,
    MultiTaskLearningAgent,
    ContinualLearningAgent,
    ExplainableRLAgent,
    AdaptiveMetaLearningAgent,
    AdvancedRLExperimentManager
)

# Package metadata
__all__ = [
    # Core modules
    'offline_rl', 'safe_rl', 'multi_agent', 'robust_rl', 
    'environments', 'evaluation', 'utils',
    
    # Advanced modules
    'advanced_algorithms', 'complex_environments', 
    'advanced_visualizations', 'advanced_concepts',
    
    # Core classes
    'ConservativeQLearning', 'ImplicitQLearning', 'OfflineDataset',
    'SafeEnvironment', 'ConstrainedPolicyOptimization', 'LagrangianSafeRL',
    'MultiAgentEnvironment', 'MADDPGAgent', 'QMIXAgent', 'MultiAgentReplayBuffer',
    'RobustEnvironment', 'DomainRandomizationAgent', 'AdversarialRobustAgent',
    'ComprehensiveEvaluator',
    
    # Advanced classes
    'HierarchicalRLAgent', 'MetaLearningAgent', 'CausalRLAgent',
    'QuantumInspiredRLAgent', 'NeurosymbolicRLAgent', 'FederatedRLAgent',
    'DynamicMultiObjectiveEnvironment', 'PartiallyObservableEnvironment',
    'ContinuousControlEnvironment', 'AdversarialEnvironment', 'EnvironmentConfig',
    'Interactive3DVisualizer', 'RealTimePerformanceMonitor', 'MultiDimensionalAnalyzer',
    'CausalGraphVisualizer', 'QuantumStateVisualizer', 'FederatedLearningDashboard',
    'AdvancedMetricsAnalyzer', 'VisualizationConfig',
    'TransferLearningAgent', 'CurriculumLearningAgent', 'MultiTaskLearningAgent',
    'ContinualLearningAgent', 'ExplainableRLAgent', 'AdaptiveMetaLearningAgent',
    'AdvancedRLExperimentManager',
    
    # Utility functions
    'generate_offline_dataset', 'collect_safe_trajectory',
    'create_evaluation_environments', 'run_comprehensive_evaluation'
]