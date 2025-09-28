"""
Utilities for Safe Reinforcement Learning

This module provides utility functions for safe RL training and evaluation.
"""


def collect_safe_trajectory(env, agent, max_steps=50):
    """Collect trajectory with safety information."""
    trajectory = []
    state = env.reset()

    for step in range(max_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        constraint_cost = info['constraint_cost']

        trajectory.append((
            state.copy(), action, reward, constraint_cost, done, log_prob
        ))

        if done:
            break

        state = next_state

    return trajectory