"""
exploration.py - Exploration Strategies for Reinforcement Learning.

This module provides various exploration strategies such as epsilon-greedy and Boltzmann
exploration, crucial for balancing exploration and exploitation in reinforcement learning
algorithms like Q-Learning and SARSA. It also includes an experiment class for comparing
the performance of different strategies.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
from collections import defaultdict
import random

# Assuming GridWorld and config are available in the src package
from .environments import GridWorld
from .agents import QLearningAgent # BoltzmannQLearning uses QLearningAgent
from .config import ExplorationConfig, SEED

class ExplorationStrategies:
    """
    A collection of static methods for common exploration strategies.
    """

    @staticmethod
    def epsilon_greedy(
        q_values: Dict[Tuple[int, int], Dict[str, float]],
        state: Tuple[int, int],
        valid_actions: List[str],
        epsilon: float,
    ) -> str:
        """
        Selects an action using the epsilon-greedy strategy.

        Args:
            q_values (Dict): The Q-value table.
            state (Tuple[int, int]): The current state.
            valid_actions (List[str]): List of actions valid in the current state.
            epsilon (float): The probability of choosing a random action.

        Returns:
            str: The chosen action.
        """
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            return ExplorationStrategies._greedy_action(q_values, state, valid_actions)

    @staticmethod
    def boltzmann_exploration(
        q_values: Dict[Tuple[int, int], Dict[str, float]],
        state: Tuple[int, int],
        valid_actions: List[str],
        temperature: float,
    ) -> str:
        """
        Selects an action using the Boltzmann (softmax) exploration strategy.

        Args:
            q_values (Dict): The Q-value table.
            state (Tuple[int, int]): The current state.
            valid_actions (List[str]): List of actions valid in the current state.
            temperature (float): The temperature parameter controlling exploration intensity.

        Returns:
            str: The chosen action.
        """
        if state not in q_values or not q_values[state] or temperature <= ExplorationConfig.TEMPERATURE_MIN:
            return random.choice(valid_actions) # Fallback if no Q-values or very low temperature

        q_vals_for_state = {a: q_values[state].get(a, 0.0) for a in valid_actions}
        
        # Ensure we don't divide by zero or have extremely small temperature leading to overflow
        if temperature <= 0:
            # Effectively greedy if temperature is zero or negative
            return ExplorationStrategies._greedy_action(q_values, state, valid_actions)

        exp_q_values = {action: np.exp(q / temperature) for action, q in q_vals_for_state.items()}
        sum_exp_q = sum(exp_q_values.values())

        if sum_exp_q == 0:
            return random.choice(valid_actions) # Fallback to random if all exp(q) are zero (e.g., very negative Q-values)

        probabilities = [exp_q_values[action] / sum_exp_q for action in valid_actions]
        return np.random.choice(valid_actions, p=probabilities)

    @staticmethod
    def _greedy_action(
        q_values: Dict[Tuple[int, int], Dict[str, float]],
        state: Tuple[int, int],
        valid_actions: List[str],
    ) -> str:
        """
        Helper to get the greedy action from Q-values.
        """
        if state not in q_values or not q_values[state]:
            return random.choice(valid_actions) # Fallback to random if no Q-values

        state_q_values = {a: q_values[state].get(a, 0.0) for a in valid_actions}
        max_q = -float('inf')
        best_actions = []

        for action, q_value in state_q_values.items():
            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)
        
        return random.choice(best_actions) # Break ties randomly

    @staticmethod
    def decay_epsilon(
        current_epsilon: float,
        episode: int,
        epsilon_decay_rate: float,
        epsilon_min: float,
    ) -> float:
        """
        Decays epsilon based on an exponential decay rate.

        Args:
            current_epsilon (float): The current epsilon value.
            episode (int): The current episode number (can be ignored for simple decay).
            epsilon_decay_rate (float): The rate at which epsilon decays.
            epsilon_min (float): The minimum value epsilon can take.

        Returns:
            float: The new, decayed epsilon value.
        """
        return max(epsilon_min, current_epsilon * epsilon_decay_rate)

    @staticmethod
    def decay_temperature(
        current_temperature: float,
        episode: int,
        temperature_decay_rate: float,
        temperature_min: float,
    ) -> float:
        """
        Decays temperature based on an exponential decay rate.

        Args:
            current_temperature (float): The current temperature value.
            episode (int): The current episode number (can be ignored for simple decay).
            temperature_decay_rate (float): The rate at which temperature decays.
            temperature_min (float): The minimum value temperature can take.

        Returns:
            float: The new, decayed temperature value.
        """
        return max(temperature_min, current_temperature * temperature_decay_rate)


class BoltzmannQLearning(QLearningAgent):
    """
    Q-Learning agent that uses Boltzmann exploration for action selection.
    """

    def __init__(
        self,
        env: GridWorld,
        alpha: float = ExplorationConfig.ALPHA,
        gamma: float = ExplorationConfig.GAMMA,
        temperature: float = ExplorationConfig.TEMPERATURE_START,
        temperature_decay: float = ExplorationConfig.TEMPERATURE_DECAY,
        temperature_min: float = ExplorationConfig.TEMPERATURE_MIN,
    ):
        """
        Initializes the Boltzmann Q-Learning agent.

        Args:
            env (GridWorld): The environment instance.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            temperature (float): Initial temperature for Boltzmann exploration.
            temperature_decay (float): Decay rate for temperature.
            temperature_min (float): Minimum value for temperature.
        """
        super().__init__(env, alpha, gamma, epsilon=0, epsilon_decay=1, epsilon_min=0) # Epsilon-related params are not used for Boltzmann
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

    def get_action(self, state: Tuple[int, int], explore: bool = True) -> str:
        """
        Selects an action using the Boltzmann exploration strategy or purely greedily.

        Args:
            state (Tuple[int, int]): The current state.
            explore (bool): If True, use Boltzmann; otherwise, use greedy policy.

        Returns:
            str: The chosen action.
        """
        if explore:
            return ExplorationStrategies.boltzmann_exploration(
                self.Q, state, self.env.get_valid_actions(state), self.temperature
            )
        else:
            return self.greedy_policy_obj.get_action(state, self.Q)

    def train(self, num_episodes: int = ExplorationConfig.NUM_EPISODES, print_every: int = ExplorationConfig.PRINT_EVERY) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Trains the Boltzmann Q-Learning agent.

        Args:
            num_episodes (int): Number of episodes to train for.
            print_every (int): Frequency of printing training progress.

        Returns:
            Dict[Tuple[int, int], Dict[str, float]]: The learned Q-value function.
        """
        print(f"Training Boltzmann Q-Learning agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action = self.get_action(state, explore=True)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q(state, action, reward, next_state, done) # Use QLearning update rule
                episode_reward += reward
                steps += 1
                state = next_state
            
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(steps)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Temperature: {self.temperature:.2f}")
        print("Boltzmann Q-Learning training complete.")
        return self.Q


class ExplorationExperiment:
    """
    Facilitates running and comparing different exploration strategies.
    """

    def __init__(self, env: GridWorld):
        """
        Initializes the ExplorationExperiment with a GridWorld environment.

        Args:
            env (GridWorld): The environment instance.
        """
        self.env = env
        self.results = defaultdict(list)

    def run_exploration_experiment(
        self,
        strategies: Dict[str, Dict[str, float]],
        num_episodes: int = ExplorationConfig.NUM_EPISODES,
        num_runs: int = ExplorationConfig.NUM_RUNS,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Runs a comparison experiment for various exploration strategies.

        Args:
            strategies (Dict[str, Dict[str, float]]): A dictionary mapping strategy names
                                                     to their parameters (epsilon or temperature).
            num_episodes (int): Number of episodes for each training run.
            num_runs (int): Number of independent runs for each strategy to average results.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary of results for each strategy, including
                                             episode rewards, evaluation metrics, and agent state.
        """
        all_results = defaultdict(list)
        for strategy_name, params in strategies.items():
            print(f"\n--- Running {strategy_name} strategy (Total Runs: {num_runs}) ---")
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...")
                np.random.seed(SEED + run) # Ensure different seeds for each run
                random.seed(SEED + run)

                if "epsilon" in params:
                    agent = QLearningAgent(
                        self.env,
                        epsilon=params["epsilon"],
                        epsilon_decay=params.get("decay", ExplorationConfig.EPSILON_DECAY),
                        alpha=ExplorationConfig.ALPHA, # Assuming AgentConfig is not defined, using ExplorationConfig
                        gamma=ExplorationConfig.GAMMA, # Assuming AgentConfig is not defined, using ExplorationConfig
                    )
                elif "temperature" in params:
                    agent = BoltzmannQLearning(
                        self.env,
                        temperature=params["temperature"],
                        temperature_decay=params.get("decay", ExplorationConfig.TEMPERATURE_DECAY),
                        alpha=ExplorationConfig.ALPHA, # Assuming AgentConfig is not defined, using ExplorationConfig
                        gamma=ExplorationConfig.GAMMA, # Assuming AgentConfig is not defined, using ExplorationConfig
                    )
                else:
                    raise ValueError(f"Unknown exploration strategy parameters for {strategy_name}")
                
                agent.train(num_episodes=num_episodes, print_every=num_episodes+1) # Suppress intermediate prints
                evaluation_results = agent.evaluate_policy(num_episodes=ExplorationConfig.EVAL_EPISODES) # Assuming ExperimentConfig is not defined, using ExplorationConfig
                
                all_results[strategy_name].append({
                    "episode_rewards": agent.episode_rewards,
                    "evaluation": evaluation_results,
                    "agent_q_values": agent.Q # Store final Q-values for analysis
                })
                print(f"  Run {run + 1}/{num_runs} completed. Avg Reward: {evaluation_results['avg_reward']:.2f}, Success Rate: {evaluation_results['success_rate']*100:.1f}%")

        self.results = all_results
        return all_results

    def analyze_exploration_results(self, results: Dict[str, List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyzes the results of the exploration experiment.

        Args:
            results (Dict): The results from run_exploration_experiment. If None, uses internal results.

        Returns:
            Dict[str, Dict[str, float]]: A summary of performance for each strategy.
        """
        if results is None:
            results = self.results

        performance_summary = {}
        for strategy_name, runs_data in results.items():
            avg_rewards = [run["evaluation"]["avg_reward"] for run in runs_data]
            avg_success_rates = [run["evaluation"]["success_rate"] for run in runs_data]
            
            performance_summary[strategy_name] = {
                "mean_reward": np.mean(avg_rewards),
                "std_reward": np.std(avg_rewards),
                "mean_success_rate": np.mean(avg_success_rates),
                "std_success_rate": np.std(avg_success_rates),
            }
            print(f"\nStrategy: {strategy_name}")
            print(f"  Mean Reward: {performance_summary[strategy_name]['mean_reward']:.2f} ± {performance_summary[strategy_name]['std_reward']:.2f}")
            print(f"  Mean Success Rate: {performance_summary[strategy_name]['mean_success_rate']*100:.1f}% ± {performance_summary[strategy_name]['std_success_rate']*100:.1f}%")
        
        return performance_summary


if __name__ == "__main__":
    # Example Usage:
    env = GridWorld()
    print("--- Exploration Strategies Demo ---")

    strategies = {
        "epsilon_0.1": {"epsilon": 0.1, "decay": 1.0},
        "epsilon_decay_fast": {"epsilon": 0.9, "decay": 0.99},
        "boltzmann_2.0": {"temperature": 2.0},
    }

    exp_runner = ExplorationExperiment(env)
    results = exp_runner.run_exploration_experiment(strategies, num_episodes=100, num_runs=3)
    exp_runner.analyze_exploration_results(results)

    print("\n--- Boltzmann Q-Learning Demo ---")
    boltzmann_agent = BoltzmannQLearning(env)
    boltzmann_agent.train(num_episodes=200)
    eval_results = boltzmann_agent.evaluate_policy()
    print(f"Boltzmann Q-Learning Avg Reward: {eval_results['avg_reward']:.2f}")
    env.visualize_values(boltzmann_agent.get_value_function(), title="Boltzmann Q-Learning Value Function", policy=boltzmann_agent.get_policy(), filepath="visualizations/boltzmann_q_values.png")

