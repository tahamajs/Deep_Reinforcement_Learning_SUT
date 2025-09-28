# Author: Taha Majlesi - 810101504, University of Tehran

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

    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))

    for state in range(env.nS):
        for action in range(env.nA):
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                env.T[state, action, nextstate] = prob
                env.R[state, action, nextstate] = reward
    return env
