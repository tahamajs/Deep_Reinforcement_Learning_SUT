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
    # Derive discrete MDP sizes (support Gymnasium/FrozenLake wrappers)
    nS = getattr(env, "nS", None)
    nA = getattr(env, "nA", None)
    if nS is None:
        nS = getattr(env.observation_space, "n", None) or (getattr(env, "T", None).shape[0] if getattr(env, "T", None) is not None else None)
    if nA is None:
        nA = getattr(env.action_space, "n", None) or (getattr(env, "T", None).shape[1] if getattr(env, "T", None) is not None else None)

    value_func = np.zeros(nS)
    next_value_func = np.zeros(nS)
    iters = 0
    delta = np.ones(nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(nS)
        for state in range(nS):

            action = policy[state]

            new_value = 0
            # Prefer dense T/R arrays if available for speed/compatibility
            if getattr(env, "T", None) is not None and getattr(env, "R", None) is not None:
                T = env.T
                R = env.R
                for nextstate in range(nS):
                    prob = T[state, action, nextstate]
                    if prob == 0:
                        continue
                    reward = R[state, action, nextstate]
                    # We don't have explicit terminal flag here; use 0
                    new_value += prob * (reward + gamma * value_func[nextstate])
            else:
                for prob, nextstate, reward, is_terminal in env.P[state][action]:
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
    nS = getattr(env, "nS", None) or getattr(env.observation_space, "n", None)
    value_func = np.zeros(nS)
    iters = 0
    delta = np.ones(nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(nS)
        for state in range(nS):

            action = policy[state]

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
    nS = getattr(env, "nS", None) or getattr(env.observation_space, "n", None)
    value_func = np.zeros(nS)
    iters = 0
    delta = np.ones(nS)
    while iters < max_iterations and np.any((delta > tol)):
        delta = np.zeros(nS)
        states = np.random.choice(nS, nS, replace=False)

        for state in states:

            action = policy[state]

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

            delta[state] = max(delta[state], abs(value_func[state] - new_value))
            value_func[state] = new_value
        iters += 1
    return value_func, iters