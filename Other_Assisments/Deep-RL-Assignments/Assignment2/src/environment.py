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
    if nS is None or nA is None or P is None:
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
