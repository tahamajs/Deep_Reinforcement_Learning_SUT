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
    # derive discrete sizes
    nS = getattr(env, "nS", None) or getattr(env.observation_space, "n", None)
    nA = getattr(env, "nA", None) or getattr(env.action_space, "n", None)
    if nS is None or nA is None:
        # try from T array if present
        if getattr(env, "T", None) is not None:
            nS = env.T.shape[0]
            nA = env.T.shape[1]
        else:
            raise RuntimeError("Cannot determine nS/nA for policy improvement")

    for state in range(nS):
        max_value = -np.inf
        best_action = -1
        for action in range(nA):
            value = 0
            # prefer T/R arrays if available
            if (
                getattr(env, "T", None) is not None
                and getattr(env, "R", None) is not None
            ):
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

        if policy[state] != best_action:
            policy_stable = False
            policy[state] = best_action
    print("Policy {}".format(policy))
    return policy_stable, policy
