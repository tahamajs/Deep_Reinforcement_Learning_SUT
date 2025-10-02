from gymnasium.envs.registration import register

register(
    id="sparse-cheetah-cs285-v1",
    entry_point="cs285.envs.sparse_half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
)
from cs285.envs.sparse_half_cheetah import HalfCheetahEnv

register(
    id="PointmassEasy-v0",
    entry_point="cs285.envs.pointmass:PointMass",
    max_episode_steps=50,
)

register(
    id="PointmassMedium-v0",
    entry_point="cs285.envs.pointmass:PointMass",
    max_episode_steps=150,
)

register(
    id="PointmassHard-v0",
    entry_point="cs285.envs.pointmass:PointMass",
    max_episode_steps=100,
)

register(
    id="PointmassVeryHard-v0",
    entry_point="cs285.envs.pointmass:PointMass",
    max_episode_steps=200,
)