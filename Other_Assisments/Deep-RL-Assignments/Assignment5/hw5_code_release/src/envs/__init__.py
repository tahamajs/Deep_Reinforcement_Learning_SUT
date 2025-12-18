"""Register the 2D pushing environments."""

from gymnasium.envs.registration import register
from .pusher_env import Pushing2DEnv, Pushing2DNoisyControlEnv

# Register the environments
register(
    id="Pushing2D-v1",
    entry_point="envs:Pushing2DEnv",
    max_episode_steps=40,
)

register(
    id="Pushing2DNoisyControl-v1",
    entry_point="envs:Pushing2DNoisyControlEnv",
    max_episode_steps=40,
)
