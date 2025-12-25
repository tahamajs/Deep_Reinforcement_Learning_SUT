from cs285.envs import obstacles

# Attempt to register other optional MuJoCo-based envs if available
try:
    from cs285.envs import cheetah, reacher
except Exception:
    # If mujoco or dependencies are missing, skip silently
    pass