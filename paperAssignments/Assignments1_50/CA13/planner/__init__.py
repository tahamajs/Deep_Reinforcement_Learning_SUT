"""SimGolf planner package exports."""

from .checkpoint_buffer import CheckpointBuffer
from .triggers import should_trigger, TriggerConfig

# distillation utilities
from .distill import branch_to_action_distribution, select_branch_action

# prefer advanced simulate_branches implementation if available
try:
    from .advanced import simulate_branches_advanced as simulate_branches, Branch
except Exception:
    from .simgolf_latent import simulate_branches, Branch  # type: ignore

__all__ = [
    "CheckpointBuffer",
    "simulate_branches",
    "Branch",
    "should_trigger",
    "TriggerConfig",
    "branch_to_action_distribution",
    "select_branch_action",
]
















