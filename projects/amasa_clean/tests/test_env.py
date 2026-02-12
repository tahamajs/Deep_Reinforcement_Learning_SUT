import numpy as np

from projects.amasa_clean.amasa.envs import SuturingEnv, make_scenario_env


def test_env_shapes_and_bounds():
    env = SuturingEnv(max_steps=20, seed=0)
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    assert env.action_space.shape[0] == 7


def test_env_determinism_fixed_seed():
    env1 = SuturingEnv(max_steps=20, seed=42)
    env2 = SuturingEnv(max_steps=20, seed=42)
    o1, _ = env1.reset(seed=42)
    o2, _ = env2.reset(seed=42)
    assert np.allclose(o1, o2)


def test_scenario_factory():
    env = make_scenario_env("adversarial", max_steps=10, seed=1)
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
