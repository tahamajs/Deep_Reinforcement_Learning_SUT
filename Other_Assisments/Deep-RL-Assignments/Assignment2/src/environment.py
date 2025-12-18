import gymnasium as gym
import numpy as np


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)
    # unwrap to the base environment implementation (Gymnasium wrappers like
    # OrderEnforcing do not expose attributes such as nS/nA/P)
    base = getattr(env, "unwrapped", env)

    nS = getattr(base, "nS", None)
    nA = getattr(base, "nA", None)
    P = getattr(base, "P", None)
    # If the base env does not expose the classic discrete-MDP attributes
    # (nS, nA, P), try fallbacks for FrozenLake-like envs or derive dims from
    # observation/action spaces.
    if nS is None or nA is None or P is None:
        # Try deriving nS/nA from discrete spaces
        obs_n = getattr(base.observation_space, "n", None)
        act_n = getattr(base.action_space, "n", None)
        if obs_n is not None and act_n is not None:
            nS = obs_n
            nA = act_n
        else:
            # Special-case FrozenLake: try to construct a fresh FrozenLakeEnv
            try:
                import deeprl_hw2q2.lake_envs as lake_envs

                if "4x4" in env_name:
                    desc = lake_envs.MAPS["4x4"]
                else:
                    desc = lake_envs.MAPS.get("8x8")
                is_slippery = "Stochastic" in env_name
                # Build a new FrozenLakeEnv instance which exposes P/nS/nA
                new_base = lake_envs.frozen_lake.FrozenLakeEnv(desc=desc, is_slippery=is_slippery)
                base = new_base
                nS = getattr(base, "nS", None)
                nA = getattr(base, "nA", None)
                P = getattr(base, "P", None)
            except Exception:
                raise RuntimeError(
                    f"Environment {env_name!r} does not expose discrete MDP attributes (nS,nA,P)."
                )

    base.T = np.zeros((nS, nA, nS))
    base.R = np.zeros((nS, nA, nS))

    for state in range(nS):
        for action in range(nA):
            for prob, nextstate, reward, is_terminal in P[state][action]:
                base.T[state, action, nextstate] = prob
                base.R[state, action, nextstate] = reward
    # return the unwrapped/base env so callers can access nS/nA/P directly
    return base
