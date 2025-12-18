import numpy as np
import gymnasium as gym
def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype("str")
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
def value_function_to_policy(env, gamma, value_func):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    # Derive sizes
    nS = getattr(env, "nS", None) or getattr(env.observation_space, "n", None)
    nA = getattr(env, "nA", None) or getattr(env.action_space, "n", None)
    if nS is None or nA is None:
        if getattr(env, "T", None) is not None:
            nS = env.T.shape[0]
            nA = env.T.shape[1]
        else:
            raise RuntimeError("Cannot determine nS/nA to derive policy from value function")

    policy = np.zeros(nS, dtype="int")
    for state in range(nS):
        max_value = -np.inf
        best_action = -1
        for action in range(nA):
            value = 0
            if getattr(env, "T", None) is not None and getattr(env, "R", None) is not None:
                T = env.T
                R = env.R
                for nextstate in range(nS):
                    prob = T[state, action, nextstate]
                    if prob == 0:
                        continue
                    reward = R[state, action, nextstate]
                    value += prob * (reward + gamma * value_func[nextstate])
            else:
                for prob, nextstate, reward, is_terminal in env.P[state][action]:
                    value += prob * (
                        reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                    )
            if max_value < value:
                max_value = value
                best_action = action

        policy[state] = best_action
    return policy