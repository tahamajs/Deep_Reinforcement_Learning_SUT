import numpy as np
from src.utils import value_function_to_policy
from src.visualization import display_policy_letters, value_func_heatmap
def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    nS = getattr(env, "nS", None) or getattr(env.observation_space, "n", None)
    nA = getattr(env, "nA", None) or getattr(env.action_space, "n", None)
    value_func = np.zeros(nS)
    next_value_func = np.zeros(nS)
    iters = 0
    delta = np.ones(nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(nS)
        for state in range(nS):
            max_value = -np.inf
            for action in range(nA):
                new_value = 0
                if getattr(env, "T", None) is not None and getattr(env, "R", None) is not None:
                    T = env.T
                    R = env.R
                    for nextstate in range(nS):
                        prob = T[state, action, nextstate]
                        if prob == 0:
                            continue
                        reward = R[state, action, nextstate]
                        new_value += prob * (reward + gamma * value_func[nextstate])
                else:
                    for prob, nextstate, reward, is_terminal in env.P[state][action]:
                        new_value += prob * (
                            reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                        )

                if max_value < new_value:
                    max_value = new_value

            delta[state] = max(delta[state], abs(value_func[state] - max_value))
            next_value_func[state] = max_value
        value_func = next_value_func.copy()
        iters += 1
    print("Policy Evaluation Complete \n{}".format(value_func))

    policy = value_function_to_policy(env, gamma, value_func)

    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    return value_func, iters
def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    nS = getattr(env, "nS", None) or getattr(env.observation_space, "n", None)
    nA = getattr(env, "nA", None) or getattr(env.action_space, "n", None)
    value_func = np.zeros(nS)
    iters = 0
    delta = np.ones(nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(nS)
        for state in range(nS):
            max_value = -np.inf
            for action in range(nA):
                new_value = 0
                if getattr(env, "T", None) is not None and getattr(env, "R", None) is not None:
                    T = env.T
                    R = env.R
                    for nextstate in range(nS):
                        prob = T[state, action, nextstate]
                        if prob == 0:
                            continue
                        reward = R[state, action, nextstate]
                        new_value += prob * (reward + gamma * value_func[nextstate])
                else:
                    for prob, nextstate, reward, is_terminal in env.P[state][action]:
                        new_value += prob * (
                            reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                        )

                if max_value < new_value:
                    max_value = new_value

            delta[state] = max(delta[state], abs(value_func[state] - max_value))
            value_func[state] = max_value
        iters += 1
    print("Policy Evaluation Complete \n{}".format(value_func))

    policy = value_function_to_policy(env, gamma, value_func)

    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    return value_func, iters
def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    nS = getattr(env, "nS", None) or getattr(env.observation_space, "n", None)
    nA = getattr(env, "nA", None) or getattr(env.action_space, "n", None)
    value_func = np.zeros(nS)
    iters = 0
    delta = np.ones(nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(nS)
        states = np.random.choice(nS, nS, replace=False)

        for state in states:
            max_value = -np.inf
            for action in range(nA):
                new_value = 0
                if getattr(env, "T", None) is not None and getattr(env, "R", None) is not None:
                    T = env.T
                    R = env.R
                    for nextstate in range(nS):
                        prob = T[state, action, nextstate]
                        if prob == 0:
                            continue
                        reward = R[state, action, nextstate]
                        new_value += prob * (reward + gamma * value_func[nextstate])
                else:
                    for prob, nextstate, reward, is_terminal in env.P[state][action]:
                        new_value += prob * (
                            reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                        )

                if max_value < new_value:
                    max_value = new_value

            delta[state] = max(delta[state], abs(value_func[state] - max_value))
            value_func[state] = max_value
        iters += 1
    print("Policy Evaluation Complete \n{}".format(value_func))

    policy = value_function_to_policy(env, gamma, value_func)

    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    return value_func, iters
def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    return value_func, 0