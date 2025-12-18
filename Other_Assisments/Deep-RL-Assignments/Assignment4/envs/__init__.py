"""Register the 2D pushing environments."""

from gymnasium.envs.registration import register
from .pusher_env import Pushing2DEnv, Pushing2DNoisyControlEnv

# Register the environments
register(
    id="Pushing2D-v0",
    entry_point="envs.pusher_env:Pushing2DEnv",
    max_episode_steps=40,
)

register(
    id="Pushing2DNoisyControl-v0",
    entry_point="envs.pusher_env:Pushing2DNoisyControlEnv",
    max_episode_steps=40,
)
