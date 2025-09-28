# Author: Taha Majlesi - 810101504, University of Tehran

import numpy as np
from src.policy_evaluation import (
    evaluate_policy_sync,
    evaluate_policy_async_ordered,
    evaluate_policy_async_randperm,
)
from src.policy_improvement import improve_policy
from src.utils import value_function_to_policy
from src.visualization import display_policy_letters, value_func_heatmap


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

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
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype="int")
    display_policy_letters(env, policy)
    policy_stable = False
    value_iters = 0
    policy_iters = 0
    iters = 0
    while not policy_stable:
        iters += 1
        value_func, i = evaluate_policy_sync(env, gamma, policy)
        value_iters += i
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        policy_iters += 1
        print(
            "iters {} | policy eval {} | policy improvement {}".format(
                iters, value_iters, policy_iters
            )
        )
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    print("Policy Evaluation Complete \n{}".format(value_func))
    return policy, value_func, policy_iters, value_iters


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

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
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype="int")
    display_policy_letters(env, policy)
    policy_stable = False
    value_iters = 0
    policy_iters = 0
    iters = 0
    while not policy_stable:
        iters += 1
        value_func, i = evaluate_policy_async_ordered(env, gamma, policy)
        value_iters += i
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        policy_iters += 1
        print(
            "iters {} | policy eval {} | policy improvement {}".format(
                iters, value_iters, policy_iters
            )
        )
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    print("Policy Evaluation Complete \n{}".format(value_func))
    return policy, value_func, policy_iters, value_iters


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

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
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype="int")
    display_policy_letters(env, policy)
    policy_stable = False
    value_iters = 0
    policy_iters = 0
    iters = 0
    while not policy_stable:
        iters += 1
        value_func, i = evaluate_policy_async_randperm(env, gamma, policy)
        value_iters += i
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        policy_iters += 1
        print(
            "iters {} | policy eval {} | policy improvement {}".format(
                iters, value_iters, policy_iters
            )
        )
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    print("Policy Evaluation Complete \n{}".format(value_func))
    return policy, value_func, policy_iters, value_iters
