"""
experiments.py - Functions for running and comparing Temporal Difference Learning experiments.

This module provides high-level functions to set up, run, and collect results
from training TD(0), Q-Learning, and SARSA agents, as well as comparing
different exploration strategies. It centralizes the experimental protocol.
"""

from typing import Dict, Tuple, List, Any
import numpy as np

# Assuming all necessary components are available in the src package
from .environments import GridWorld
from .agents import TD0Agent, QLearningAgent, SARSAAgent, RandomPolicy, BasePolicy
from .exploration import ExplorationStrategies, BoltzmannQLearning, ExplorationExperiment
from .config import AgentConfig, ExperimentConfig, SEED


def experiment_td0(
    env: GridWorld,
    policy: BasePolicy,
    num_episodes: int = AgentConfig.NUM_EPISODES,
    alpha: float = AgentConfig.ALPHA,
    gamma: float = AgentConfig.GAMMA,
) -> Tuple[TD0Agent, Dict[Tuple[int, int], float]]:
    """
    Runs an experiment for the TD(0) agent to evaluate a given policy.

    Args:
        env (GridWorld): The environment instance.
        policy (BasePolicy): The policy to evaluate (e.g., RandomPolicy).
        num_episodes (int): Number of episodes to train the agent.
        alpha (float): Learning rate.
        gamma (float): Discount factor.

    Returns:
        Tuple[TD0Agent, Dict[Tuple[int, int], float]]: The trained TD(0) agent and its learned V-function.
    """
    print(f"\n--- Running TD(0) Experiment ({num_episodes} episodes) ---")
    agent = TD0Agent(env, policy, alpha=alpha, gamma=gamma)
    V_td = agent.train(num_episodes=num_episodes, print_every=AgentConfig.PRINT_EVERY)
    print("TD(0) Experiment Completed.")
    return agent, V_td


def experiment_q_learning(
    env: GridWorld,
    num_episodes: int = AgentConfig.NUM_EPISODES,
    alpha: float = AgentConfig.ALPHA,
    gamma: float = AgentConfig.GAMMA,
    epsilon: float = ExplorationConfig.EPSILON_START,
    epsilon_decay: float = ExplorationConfig.EPSILON_DECAY,
    epsilon_min: float = ExplorationConfig.EPSILON_MIN,
) -> Tuple[QLearningAgent, Dict[Tuple[int, int], float], Dict[Tuple[int, int], str], Dict[str, Any]]:
    """
    Runs an experiment for the Q-Learning agent to learn an optimal policy.

    Args:
        env (GridWorld): The environment instance.
        num_episodes (int): Number of episodes to train the agent.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        epsilon_min (float): Minimum epsilon value.

    Returns:
        Tuple[QLearningAgent, Dict[Tuple[int, int], float], Dict[Tuple[int, int], str], Dict[str, Any]]:
            - The trained Q-Learning agent.
            - The extracted optimal V-function.
            - The extracted optimal policy.
            - Evaluation metrics of the optimal policy.
    """
    print(f"\n--- Running Q-Learning Experiment ({num_episodes} episodes) ---")
    agent = QLearningAgent(
        env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )
    agent.train(num_episodes=num_episodes, print_every=AgentConfig.PRINT_EVERY)
    V_optimal = agent.get_value_function()
    optimal_policy = agent.get_policy()
    evaluation = agent.evaluate_policy(num_episodes=ExperimentConfig.EVAL_EPISODES)
    print(f"Q-Learning Experiment Completed. Success Rate: {evaluation['success_rate']*100:.1f}%")
    return agent, V_optimal, optimal_policy, evaluation


def experiment_sarsa(
    env: GridWorld,
    num_episodes: int = AgentConfig.NUM_EPISODES,
    alpha: float = AgentConfig.ALPHA,
    gamma: float = AgentConfig.GAMMA,
    epsilon: float = ExplorationConfig.EPSILON_START,
    epsilon_decay: float = ExplorationConfig.EPSILON_DECAY,
    epsilon_min: float = ExplorationConfig.EPSILON_MIN,
) -> Tuple[SARSAAgent, Dict[Tuple[int, int], float], Dict[Tuple[int, int], str], Dict[str, Any]]:
    """
    Runs an experiment for the SARSA agent to learn an on-policy control policy.

    Args:
        env (GridWorld): The environment instance.
        num_episodes (int): Number of episodes to train the agent.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        epsilon_min (float): Minimum epsilon value.

    Returns:
        Tuple[SARSAAgent, Dict[Tuple[int, int], float], Dict[Tuple[int, int], str], Dict[str, Any]]:
            - The trained SARSA agent.
            - The extracted V-function for the learned policy.
            - The extracted policy.
            - Evaluation metrics of the policy.
    """
    print(f"\n--- Running SARSA Experiment ({num_episodes} episodes) ---")
    agent = SARSAAgent(
        env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )
    agent.train(num_episodes=num_episodes, print_every=AgentConfig.PRINT_EVERY)
    V_sarsa = agent.get_value_function()
    sarsa_policy = agent.get_policy()
    evaluation = agent.evaluate_policy(num_episodes=ExperimentConfig.EVAL_EPISODES)
    print(f"SARSA Experiment Completed. Success Rate: {evaluation['success_rate']*100:.1f}%")
    return agent, V_sarsa, sarsa_policy, evaluation


def experiment_exploration_strategies(
    env: GridWorld,
    strategies: Dict[str, Dict[str, float]],
    num_episodes: int = AgentConfig.NUM_EPISODES,
    num_runs: int = ExperimentConfig.NUM_RUNS,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs a comparison experiment for various exploration strategies using Q-Learning.

    Args:
        env (GridWorld): The environment instance.
        strategies (Dict[str, Dict[str, float]]): A dictionary mapping strategy names
                                                 to their parameters (epsilon or temperature).
        num_episodes (int): Number of episodes for each training run.
        num_runs (int): Number of independent runs for each strategy to average results.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary of results for each strategy, including
                                         episode rewards, evaluation metrics, and agent state.
    """
    print(f"\n--- Running Exploration Strategies Comparison (Episodes: {num_episodes}, Runs: {num_runs}) ---")
    exp_runner = ExplorationExperiment(env)
    results = exp_runner.run_exploration_experiment(strategies, num_episodes, num_runs)
    print("Exploration Strategies Comparison Completed.")
    exp_runner.analyze_exploration_results(results) # Print summary during experiment
    return results


if __name__ == "__main__":
    # Example Usage of experiment functions:
    env = GridWorld()
    random_policy = RandomPolicy(env)
    
    # Run TD(0)
    td_agent, V_td = experiment_td0(env, random_policy, num_episodes=100)

    # Run Q-Learning
    q_agent, V_optimal, q_policy, q_eval = experiment_q_learning(env, num_episodes=200)

    # Run SARSA
    sarsa_agent, V_sarsa, sarsa_policy, sarsa_eval = experiment_sarsa(env, num_episodes=200)

    # Run Exploration Strategies Comparison
    strategies_to_test = {
        "epsilon_0.1": {"epsilon": 0.1, "decay": 1.0},
        "epsilon_decay": {"epsilon": 0.9, "decay": 0.99},
        "boltzmann_2.0": {"temperature": 2.0},
    }
    exploration_results = experiment_exploration_strategies(env, strategies_to_test, num_episodes=50, num_runs=2)
    print("\nAll example experiments finished.")
