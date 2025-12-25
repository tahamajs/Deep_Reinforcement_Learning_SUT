"""Custom environments for HW4 with compatibility patches for newer Gym/MuJoCo."""

# Compatibility shim: older envs instantiate MujocoEnv with signature
# MujocoEnv(model_path, frame_skip) but newer gym versions require
# an observation_space argument. We monkey-patch MujocoEnv.__init__ to
# accept the old signature by injecting a dummy observation_space when
# it's not provided.
try:
    import gym
    from gym.envs.mujoco import MujocoEnv as _MujocoEnv
    import numpy as _np
    from gym.spaces import Box as _Box

    _orig_mujoco_init = _MujocoEnv.__init__

    def _compat_mujoco_init(self, model_path, frame_skip, *args, **kwargs):
        # If observation_space provided already, call original
        if 'observation_space' in kwargs or (len(args) > 0 and isinstance(args[0], gym.spaces.Space)):
            return _orig_mujoco_init(self, model_path, frame_skip, *args, **kwargs)
        # Inject a small dummy observation space to maintain backwards compatibility
        dummy = _Box(low=-_np.inf, high=_np.inf, shape=(1,), dtype=_np.float64)
        return _orig_mujoco_init(self, model_path, frame_skip, observation_space=dummy, *args, **kwargs)

    _MujocoEnv.__init__ = _compat_mujoco_init
except Exception:
    # If gym/mujoco not available or something fails, continue silently; imports below will fail later if needed
    pass


def register_envs():
    """Register custom environments with Gym."""
    try:
        # Import submodules to trigger their registration
        # Keep this best-effort so imports won't crash if mujoco/etc are missing
        from cs285.envs import cheetah, reacher, obstacles
    except ImportError:
        # Fall back to importing individually to give clearer partial availability
        try:
            from cs285.envs import cheetah
        except Exception:
            pass
        try:
            from cs285.envs import reacher
        except Exception:
            pass
        try:
            from cs285.envs import obstacles
        except Exception:
            pass


# Attempt to register envs upon import; silent on failure
try:
    register_envs()
except Exception:
    pass
