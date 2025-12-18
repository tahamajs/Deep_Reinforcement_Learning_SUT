"""SimGolf planner package exports."""

from .checkpoint_buffer import CheckpointBuffer

# prefer advanced simulate_branches implementation if available
try:
    from .advanced import simulate_branches_advanced as simulate_branches, Branch
except Exception:
    from .simgolf_latent import simulate_branches, Branch  # type: ignore

from .triggers import should_trigger

__all__ = ["CheckpointBuffer", "simulate_branches", "Branch", "should_trigger"]













