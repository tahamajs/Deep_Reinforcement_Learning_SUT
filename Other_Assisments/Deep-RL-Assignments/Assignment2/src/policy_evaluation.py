# Author: Taha Majlesi - 810101504, University of Tehran

import numpy as np


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    next_value_func = np.zeros(env.nS)
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(env.nS)
        for state in range(env.nS):
            # Find out the current action encoded in the policy
            action = policy[state]
            # Iterate over all the future states
            new_value = 0
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                prob = env.T[state, action, nextstate]
                reward = env.R[state, action, nextstate]
                new_value += prob * (
                    reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                )

            delta[state] = max(delta[state], abs(next_value_func[state] - new_value))
            next_value_func[state] = new_value
        value_func = next_value_func.copy()
        iters += 1
    return value_func, iters


def evaluate_policy_async_ordered(
    env, gamma, policy, max_iterations=int(1e3), tol=1e-3
):
    """Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(env.nS)
        for state in range(env.nS):
            # Find out the current action encoded in the policy
            action = policy[state]
            # Iterate over all the future states
            new_value = 0
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                prob = env.T[state, action, nextstate]
                reward = env.R[state, action, nextstate]
                new_value += prob * (
                    reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                )

            delta[state] = max(delta[state], abs(value_func[state] - new_value))
            value_func[state] = new_value
        iters += 1
    return value_func, iters


def evaluate_policy_async_randperm(
    env, gamma, policy, max_iterations=int(1e3), tol=1e-3
):
    """Performs policy evaluation.

    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(env.nS)
        states = np.random.choice(env.nS, env.nS, replace=False)

        for state in states:
            # Find out the current action encoded in the policy
            action = policy[state]
            # Iterate over all the future states
            new_value = 0
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                prob = env.T[state, action, nextstate]
                reward = env.R[state, action, nextstate]
                new_value += prob * (
                    reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                )

            delta[state] = max(delta[state], abs(value_func[state] - new_value))
            value_func[state] = new_value
        iters += 1
    return value_func, iters
