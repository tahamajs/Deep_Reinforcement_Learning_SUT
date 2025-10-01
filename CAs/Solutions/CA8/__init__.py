"""
CA8: Causal Reasoning and Multi-Modal Reinforcement Learning
"""

from .utils.causal_rl_utils import device
from .agents.causal_discovery import CausalGraph, CausalDiscovery
from .agents.causal_rl_agent import (
    CausalRLAgent,
    CounterfactualRLAgent,
    CausalReasoningNetwork,
)
from .environments.multi_modal_env import (
    MultiModalGridWorld,
    MultiModalWrapper,
    PromptTemplate,
)
from .evaluation.metrics import (
    CausalDiscoveryMetrics,
    MultiModalMetrics,
    CausalRLMetrics,
    IntegratedMetrics,
)
from .experiments.causal_experiments import CausalDiscoveryExperiments
from .experiments.multimodal_experiments import MultiModalExperiments
from .experiments.integrated_experiments import IntegratedExperiments
from .models.fusion_networks import (
    EarlyFusionNetwork,
    LateFusionNetwork,
    CrossModalAttentionNetwork,
)

__version__ = "1.0.0"
__all__ = [
    "device",
    "CausalGraph",
    "CausalDiscovery",
    "CausalRLAgent",
    "CounterfactualRLAgent",
    "CausalReasoningNetwork",
    "MultiModalGridWorld",
    "MultiModalWrapper",
    "PromptTemplate",
    "CausalDiscoveryMetrics",
    "MultiModalMetrics",
    "CausalRLMetrics",
    "IntegratedMetrics",
    "CausalDiscoveryExperiments",
    "MultiModalExperiments",
    "IntegratedExperiments",
    "EarlyFusionNetwork",
    "LateFusionNetwork",
    "CrossModalAttentionNetwork",
]
