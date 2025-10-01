"""
CA8 Models Module
Neural network models for causal multi-modal RL
"""

from .fusion_networks import (
    EarlyFusionNetwork,
    LateFusionNetwork,
    CrossModalAttentionNetwork,
    HierarchicalFusionNetwork,
)

__all__ = [
    "EarlyFusionNetwork",
    "LateFusionNetwork",
    "CrossModalAttentionNetwork",
    "HierarchicalFusionNetwork",
]
