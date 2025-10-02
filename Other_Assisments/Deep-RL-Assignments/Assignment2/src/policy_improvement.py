import numpy as np
def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    for state in range(env.nS):
        max_value = -np.inf
        best_action = -1
        for action in range(env.nA):
            value = 0
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                prob = env.T[state, action, nextstate]
                reward = env.R[state, action, nextstate]
                value += prob * (
                    reward + gamma * (1 - int(is_terminal)) * value_func[nextstate]
                )
            if max_value < value:
                max_value = value
                best_action = action

        if policy[state] != best_action:
            policy_stable = False
            policy[state] = best_action
    print("Policy {}".format(policy))
    return policy_stable, policy