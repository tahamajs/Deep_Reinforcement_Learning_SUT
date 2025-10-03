import numpy as np
from typing import Dict, List, Tuple, Optional


def experiment_td0(env, policy, num_episodes=1000, alpha=0.1, gamma=0.9):
    """Run TD(0) policy evaluation experiment"""
    from agents.algorithms import TD0Agent

    agent = TD0Agent(env, policy, alpha=alpha, gamma=gamma)
    V_td = agent.train(num_episodes=num_episodes, print_every=200)

    return agent, V_td


def experiment_q_learning(
    env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995
):
    """Run Q-Learning experiment"""
    from agents.algorithms import QLearningAgent

    agent = QLearningAgent(
        env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay
    )
    agent.train(num_episodes=num_episodes, print_every=200)

    V_optimal = agent.get_value_function()
    optimal_policy = agent.get_policy()
    evaluation = agent.evaluate_policy(num_episodes=100)

    return agent, V_optimal, optimal_policy, evaluation


def experiment_sarsa(
    env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995
):
    """Run SARSA experiment"""
    from agents.algorithms import SARSAAgent

    agent = SARSAAgent(
        env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay
    )
    agent.train(num_episodes=num_episodes, print_every=200)

    V_sarsa = agent.get_value_function()
    sarsa_policy = agent.get_policy()
    evaluation = agent.evaluate_policy(num_episodes=100)

    return agent, V_sarsa, sarsa_policy, evaluation


def experiment_exploration_strategies(env, strategies, num_episodes=300, num_runs=2):
    """Run exploration strategies comparison experiment"""
    # This function is implemented in the notebook directly
    # Return empty results for now
    return {}
