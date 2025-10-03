"""
Demonstration module for CA8: Causal Reasoning and Multi-Modal Reinforcement Learning
"""

from .causal_demonstrations import (
    demonstrate_causal_graph,
    demonstrate_causal_discovery,
    demonstrate_causal_rl,
)

from .multimodal_demonstrations import (
    demonstrate_multi_modal_env,
    demonstrate_integrated_system,
)

from .comprehensive_demonstrations import (
    run_comprehensive_experiments,
)

__all__ = [
    "demonstrate_causal_graph",
    "demonstrate_causal_discovery",
    "demonstrate_causal_rl",
    "demonstrate_multi_modal_env",
    "demonstrate_integrated_system",
    "run_comprehensive_experiments",
]
