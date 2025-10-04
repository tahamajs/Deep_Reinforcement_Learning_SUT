"""Custom environments for HW4."""

def register_envs():
    """Register custom environments with Gym."""
    try:
        # Import submodules to trigger their registration
        from cs285.envs import cheetah
    except ImportError:
        pass
