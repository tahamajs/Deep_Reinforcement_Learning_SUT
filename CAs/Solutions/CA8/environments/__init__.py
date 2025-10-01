"""
CA8 Environments Module
Multi-modal environments for causal reinforcement learning
"""

from .multi_modal_env import (
    MultiModalGridWorld,
    MultiModalWrapper,
    PromptTemplate,
)

__all__ = [
    "MultiModalGridWorld",
    "MultiModalWrapper",
    "PromptTemplate",
]

