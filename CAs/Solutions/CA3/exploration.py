import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from algorithms import QLearningAgent


class ExplorationStrategies:
    """Collection of exploration strategies for RL agents"""

    @staticmethod
    def epsilon_greedy(Q, state, valid_actions, epsilon):
        """Standard ε-greedy exploration"""
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        else:
            q_values = {action: Q[state][action] for action in valid_actions}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions)

    @staticmethod
    def boltzmann_exploration(Q, state, valid_actions, temperature):
        """Boltzmann (softmax) exploration"""
        if temperature <= 0:
            q_values = {action: Q[state][action] for action in valid_actions}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions)

        q_values = np.array([Q[state][action] for action in valid_actions])
        exp_q = np.exp(q_values / temperature)
        probabilities = exp_q / np.sum(exp_q)

        return np.random.choice(valid_actions, p=probabilities)

    @staticmethod
    def decay_epsilon(
        initial_epsilon, episode, decay_rate, min_epsilon, decay_type="exponential"
    ):
        """Different epsilon decay strategies"""
        if decay_type == "exponential":
            return max(min_epsilon, initial_epsilon * (decay_rate**episode))
        elif decay_type == "linear":
            return max(min_epsilon, initial_epsilon - decay_rate * episode)
        elif decay_type == "inverse":
            return max(min_epsilon, initial_epsilon / (1 + decay_rate * episode))
        else:
            return initial_epsilon


class ExplorationExperiment:
    """Experiment with different exploration strategies"""

    def __init__(self, env):
        self.env = env

    def run_exploration_experiment(self, strategies, num_episodes=500, num_runs=3):
        """Compare different exploration strategies"""
        results = {}

        for strategy_name, params in strategies.items():
            print(f"Testing {strategy_name}...")

            strategy_results = []
            for run in range(num_runs):
                if strategy_name.startswith("epsilon"):
                    agent = QLearningAgent(
                        self.env,
                        alpha=0.1,
                        gamma=0.9,
                        epsilon=params["epsilon"],
                        epsilon_decay=params.get("decay", 0.995),
                    )
                    agent.train(num_episodes=num_episodes, print_every=num_episodes)
                elif strategy_name == "boltzmann":
                    agent = BoltzmannQLearning(
                        self.env,
                        alpha=0.1,
                        gamma=0.9,
                        temperature=params["temperature"],
                    )
                    agent.train(num_episodes=num_episodes, print_every=num_episodes)

                evaluation = agent.evaluate_policy(num_episodes=100)
                strategy_results.append(
                    {
                        "rewards": agent.episode_rewards,
                        "evaluation": evaluation,
                        "final_epsilon": getattr(agent, "epsilon", None),
                    }
                )

            results[strategy_name] = strategy_results

        return results


class BoltzmannQLearning:
    """Q-Learning with Boltzmann exploration"""

    def __init__(
        self, env, alpha=0.1, gamma=0.9, temperature=1.0, temp_decay=0.99, min_temp=0.01
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.temp_decay = temp_decay
        self.min_temp = min_temp

        self.Q = defaultdict(lambda: defaultdict(float))
        self.episode_rewards = []
        self.temperature_history = []

    def get_action(self, state, explore=True):
        """Boltzmann action selection"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None

        if not explore:
            q_values = {action: self.Q[state][action] for action in valid_actions}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions)

        return ExplorationStrategies.boltzmann_exploration(
            self.Q, state, valid_actions, self.temperature
        )

    def train(self, num_episodes=1000, print_every=100):
        """Train with Boltzmann exploration"""
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            while steps < 200:
                action = self.get_action(state, explore=True)
                if action is None:
                    break

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                current_q = self.Q[state][action]
                if done:
                    td_target = reward
                else:
                    valid_next_actions = self.env.get_valid_actions(next_state)
                    if valid_next_actions:
                        max_next_q = max(
                            [self.Q[next_state][a] for a in valid_next_actions]
                        )
                    else:
                        max_next_q = 0.0
                    td_target = reward + self.gamma * max_next_q

                self.Q[state][action] += self.alpha * (td_target - current_q)

                state = next_state
                steps += 1

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.temperature_history.append(self.temperature)

            self.temperature = max(self.min_temp, self.temperature * self.temp_decay)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                print(
                    f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Temp = {self.temperature:.3f}"
                )

    def evaluate_policy(self, num_episodes=100):
        """Evaluate learned policy"""
        rewards = []
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            while steps < 200:
                action = self.get_action(state, explore=False)
                if action is None:
                    break

                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            rewards.append(episode_reward)

        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": sum(1 for r in rewards if r > 5) / len(rewards),
        }


def analyze_exploration_results(results):
    """Analyze and visualize exploration experiment results"""

    print("\nEXPLORATION STRATEGY COMPARISON")
    print("-" * 60)
    print(
        f"{'Strategy':<20} {'Avg Reward':<12} {'Success Rate':<15} {'Std Reward':<12}"
    )
    print("-" * 60)

    strategy_performance = {}

    for strategy, runs in results.items():
        avg_rewards = [run["evaluation"]["avg_reward"] for run in runs]
        success_rates = [run["evaluation"]["success_rate"] for run in runs]

        mean_reward = np.mean(avg_rewards)
        mean_success = np.mean(success_rates)
        std_reward = np.std(avg_rewards)

        strategy_performance[strategy] = {
            "mean_reward": mean_reward,
            "mean_success": mean_success,
            "std_reward": std_reward,
        }

        print(
            f"{strategy:<20} {mean_reward:<12.2f} {mean_success*100:<15.1f}% {std_reward:<12.3f}"
        )

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for strategy, runs in results.items():
        avg_rewards = np.mean([run["rewards"] for run in runs], axis=0)
        plt.plot(avg_rewards, label=strategy, alpha=0.8)

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Learning Curves by Exploration Strategy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    strategies_list = list(strategy_performance.keys())
    rewards = [strategy_performance[s]["mean_reward"] for s in strategies_list]
    errors = [strategy_performance[s]["std_reward"] for s in strategies_list]

    bars = plt.bar(
        range(len(strategies_list)),
        rewards,
        yerr=errors,
        capsize=5,
        alpha=0.7,
        color=["blue", "red", "green", "orange", "purple"],
    )
    plt.xticks(range(len(strategies_list)), strategies_list, rotation=45)
    plt.ylabel("Average Reward")
    plt.title("Final Performance Comparison")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    success_rates = [
        strategy_performance[s]["mean_success"] * 100 for s in strategies_list
    ]
    plt.bar(range(len(strategies_list)), success_rates, alpha=0.7, color="green")
    plt.xticks(range(len(strategies_list)), strategies_list, rotation=45)
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate Comparison")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    for strategy, runs in results.items():
        if "epsilon" in strategy and hasattr(runs[0], "final_epsilon"):
            pass

    plt.xlabel("Episode")
    plt.ylabel("Exploration Parameter")
    plt.title("Exploration Parameter Evolution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return strategy_performance
