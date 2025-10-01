"""
Reinforcement Learning Algorithms

This module contains core RL algorithms including policy evaluation,
policy improvement, and policy iteration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import copy
from .policies import RandomPolicy, GreedyActionPolicy

np.random.seed(42)


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Evaluate a policy by computing the value function.

    Args:
        env: The environment (GridWorld)
        policy: The policy to evaluate
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Dictionary mapping states to values
    """
    V = defaultdict(float)

    for iteration in range(max_iterations):
        delta = 0

        for state in env.states:
            if state == env.goal_state or state in env.obstacles:
                continue

            v_old = V[state]
            v_new = 0

            action_probs = policy.get_action_probabilities(state)

            for action, prob in action_probs.items():

                transitions = env.P[state][action]

                expected_value = 0
                for trans_prob, next_state, reward in transitions:
                    expected_value += trans_prob * (reward + gamma * V[next_state])

                v_new += prob * expected_value

            V[state] = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            print(f"Policy evaluation converged after {iteration + 1} iterations")
            break

    return V


def compute_q_from_v(env, V, gamma=0.9):
    """
    Compute Q-values from value function.

    Q(s,a) = sum over s' [P(s'|s,a) * (R(s,a,s') + gamma * V(s'))]

    Args:
        env: The environment
        V: Value function (dict)
        gamma: Discount factor

    Returns:
        Dictionary mapping (state, action) pairs to Q-values
    """
    Q = defaultdict(float)

    for state in env.states:
        if state == env.goal_state or state in env.obstacles:
            continue

        for action in env.get_valid_actions(state):
            transitions = env.P[state][action]

            q_value = 0
            for prob, next_state, reward in transitions:
                q_value += prob * (reward + gamma * V[next_state])

            Q[(state, action)] = q_value

    return Q


def compute_v_from_q(Q, env):
    """
    Compute value function from Q-values.

    V(s) = max over a Q(s,a)

    Args:
        Q: Q-value function (dict)
        env: The environment

    Returns:
        Dictionary mapping states to values
    """
    V = defaultdict(float)

    for state in env.states:
        if state == env.goal_state or state in env.obstacles:
            continue

        valid_actions = env.get_valid_actions(state)
        if valid_actions:
            V[state] = max(Q[(state, action)] for action in valid_actions)

    return V


def policy_improvement(env, V, gamma=0.9):
    """
    Improve a policy given a value function.

    Args:
        env: The environment
        V: Value function
        gamma: Discount factor

    Returns:
        Improved policy (GreedyActionPolicy)
    """
    return GreedyActionPolicy(env, V, gamma)


def policy_iteration(
    env, gamma=0.9, theta=1e-6, max_iterations=100, initial_policy=None
):
    """
    Policy Iteration algorithm.

    Args:
        env: The environment
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum iterations
        initial_policy: Starting policy (default: random)

    Returns:
        Tuple of (optimal_policy, optimal_values, iteration_history)
    """
    if initial_policy is None:
        policy = RandomPolicy(env)
    else:
        policy = initial_policy

    iteration_history = []

    for iteration in range(max_iterations):
        print(f"Policy Iteration - Iteration {iteration + 1}")

        V = policy_evaluation(env, policy, gamma, theta)

        new_policy = policy_improvement(env, V, gamma)

        policies_stable = True
        for state in env.states:
            if state == env.goal_state or state in env.obstacles:
                continue

            old_action = policy.get_action(state)
            new_action = new_policy.get_action(state)

            if old_action != new_action:
                policies_stable = False
                break

        iteration_history.append(
            {
                "iteration": iteration + 1,
                "policy": copy.deepcopy(policy),
                "values": V.copy(),
                "stable": policies_stable,
            }
        )

        if policies_stable:
            print(f"Policy iteration converged after {iteration + 1} iterations")
            break

        policy = new_policy

    return policy, V, iteration_history


def value_iteration(env, gamma=0.9, theta=1e-6, max_iterations=100):
    """
    Value Iteration algorithm.

    Args:
        env: The environment
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum iterations

    Returns:
        Tuple of (optimal_values, optimal_policy, iteration_history)
    """
    V = defaultdict(float)
    iteration_history = []

    for iteration in range(max_iterations):
        delta = 0

        for state in env.states:
            if state == env.goal_state or state in env.obstacles:
                continue

            v_old = V[state]
            v_new = float("-inf")

            for action in env.get_valid_actions(state):
                expected_value = 0
                transitions = env.P[state][action]

                for prob, next_state, reward in transitions:
                    expected_value += prob * (reward + gamma * V[next_state])

                v_new = max(v_new, expected_value)

            V[state] = v_new
            delta = max(delta, abs(v_old - v_new))

        iteration_history.append({"iteration": iteration + 1, "values": V.copy()})

        if delta < theta:
            print(f"Value iteration converged after {iteration + 1} iterations")
            break

    optimal_policy = GreedyActionPolicy(env, V, gamma)

    return V, optimal_policy, iteration_history


def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-Learning algorithm.

    Args:
        env: The environment
        num_episodes: Number of episodes to run
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Tuple of (Q_values, episode_rewards)
    """
    Q = defaultdict(float)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.start_state
        episode_reward = 0
        done = False

        while not done:
            # Check if current state is terminal
            if state == env.goal_state or state in env.obstacles:
                done = True
                break

            # Epsilon-greedy action selection
            valid_actions = env.get_valid_actions(state)
            if np.random.random() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                action = max(valid_actions, key=lambda a: Q[(state, a)])

            # Take action
            transitions = env.P[state][action]
            prob, next_state, reward = transitions[0]  # Deterministic for now

            episode_reward += reward

            # Q-learning update
            best_next_q = max(
                [Q[(next_state, a)] for a in env.get_valid_actions(next_state)] + [0]
            )
            Q[(state, action)] += alpha * (
                reward + gamma * best_next_q - Q[(state, action)]
            )

            state = next_state

            if state == env.goal_state:
                done = True

        episode_rewards.append(episode_reward)

    return Q, episode_rewards
