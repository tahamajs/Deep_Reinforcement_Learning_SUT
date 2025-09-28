"""
Experiment utilities for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple


try:
    from .environments import EnvironmentWrapper
    from .algorithms import REINFORCEAgent, ActorCriticAgent, compare_algorithms
    from .visualization import TrainingVisualizer, print_training_comparison
    from .exploration import ExplorationScheduler, EntropyBonusExploration
except ImportError:
    from environments import EnvironmentWrapper
    from algorithms import REINFORCEAgent, ActorCriticAgent, compare_algorithms
    from visualization import TrainingVisualizer, print_training_comparison
    from exploration import ExplorationScheduler, EntropyBonusExploration

import time


class PolicyGradientExperiment:
    """Comprehensive experiment runner for policy gradient methods"""

    def __init__(self, env_name: str = "CartPole-v1", **env_kwargs):
        """Initialize experiment

        Args:
            env_name: Environment name
            **env_kwargs: Environment parameters
        """
        self.env_name = env_name
        self.env = EnvironmentWrapper(env_name, **env_kwargs)
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.is_continuous = self.env.is_continuous

        self.results = {}
        self.experiment_log = []

    def run_reinforce_experiment(
        self,
        num_episodes: int = 500,
        lr: float = 0.001,
        gamma: float = 0.99,
        baseline: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run REINFORCE experiment

        Args:
            num_episodes: Number of training episodes
            lr: Learning rate
            gamma: Discount factor
            baseline: Whether to use baseline
            **kwargs: Additional parameters

        Returns:
            Experiment results
        """
        print(f"Running REINFORCE Experiment on {self.env_name}")
        print(f"Baseline: {baseline}, LR: {lr}, Episodes: {num_episodes}")
        print("-" * 50)

        agent = REINFORCEAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            lr=lr,
            gamma=gamma,
            baseline=baseline,
        )

        start_time = time.time()
        results = agent.train(self.env.env, num_episodes, print_every=50)
        training_time = time.time() - start_time

        results["training_time"] = training_time
        results["baseline_used"] = baseline
        results["learning_rate"] = lr

        self.results["reinforce"] = results
        self._log_experiment("reinforce", results)

        return results

    def run_actor_critic_experiment(
        self,
        num_episodes: int = 500,
        lr_actor: float = 0.001,
        lr_critic: float = 0.005,
        gamma: float = 0.99,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run Actor-Critic experiment

        Args:
            num_episodes: Number of training episodes
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            **kwargs: Additional parameters

        Returns:
            Experiment results
        """
        print(f"Running Actor-Critic Experiment on {self.env_name}")
        print(f"Actor LR: {lr_actor}, Critic LR: {lr_critic}, Episodes: {num_episodes}")
        print("-" * 50)

        agent = ActorCriticAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
        )

        start_time = time.time()
        results = agent.train(self.env.env, num_episodes, print_every=50)
        training_time = time.time() - start_time

        results["training_time"] = training_time
        results["actor_lr"] = lr_actor
        results["critic_lr"] = lr_critic

        self.results["actor_critic"] = results
        self._log_experiment("actor_critic", results)

        return results

    def run_comparison_experiment(
        self,
        algorithms: List[str] = ["reinforce", "actor_critic"],
        num_episodes: int = 300,
        **kwargs,
    ) -> Dict[str, Dict]:
        """Run comparison between algorithms

        Args:
            algorithms: List of algorithms to compare
            num_episodes: Episodes per algorithm
            **kwargs: Algorithm parameters

        Returns:
            Comparison results
        """
        print(f"Running Algorithm Comparison on {self.env_name}")
        print(f"Algorithms: {algorithms}")
        print(f"Episodes per algorithm: {num_episodes}")
        print("=" * 60)

        results = compare_algorithms(
            self.env.env,
            algorithms,
            self.state_size,
            self.action_size,
            num_episodes,
            **kwargs,
        )

        self.results.update(results)
        for alg_name, alg_results in results.items():
            self._log_experiment(alg_name, alg_results)

        return results

    def run_exploration_experiment(
        self,
        base_algorithm: str = "reinforce",
        exploration_strategies: List[str] = ["boltzmann", "epsilon_greedy"],
        num_episodes: int = 300,
        **kwargs,
    ) -> Dict[str, Dict]:
        """Run exploration strategy comparison

        Args:
            base_algorithm: Base algorithm to use
            exploration_strategies: Exploration strategies to test
            num_episodes: Episodes per strategy
            **kwargs: Additional parameters

        Returns:
            Exploration experiment results
        """
        print(f"Running Exploration Experiment with {base_algorithm}")
        print(f"Strategies: {exploration_strategies}")
        print("=" * 50)

        results = {}

        for strategy in exploration_strategies:
            print(f"\\nTesting {strategy} exploration...")

            exploration_scheduler = ExplorationScheduler(strategy, **kwargs)

            if base_algorithm == "reinforce":
                agent = REINFORCEAgent(self.state_size, self.action_size, **kwargs)
            else:
                agent = ActorCriticAgent(self.state_size, self.action_size, **kwargs)

            scores = []
            exploration_rates = []

            for episode in range(num_episodes):

                state = self.env.reset()
                if isinstance(state, tuple):
                    state = state[0]

                episode_reward = 0
                done = False
                truncated = False

                while not (done or truncated):

                    if hasattr(agent, "policy_net"):
                        action_probs = (
                            agent.policy_net.get_action_probs(
                                torch.FloatTensor(state).unsqueeze(0)
                            )
                            .detach()
                            .numpy()
                            .flatten()
                        )
                    else:

                        action_probs = (
                            agent.actor.get_action_probs(
                                torch.FloatTensor(state).unsqueeze(0)
                            )
                            .detach()
                            .numpy()
                            .flatten()
                        )

                    action = exploration_scheduler.select_action(action_probs)

                    next_state, reward, done, truncated, _ = self.env.step(action)

                    if base_algorithm == "reinforce":
                        log_prob = agent.policy_net.get_log_prob(
                            torch.FloatTensor(state).unsqueeze(0), action
                        )
                        agent.store_transition(state, action, reward, log_prob)
                    else:

                        _, log_prob, value = agent.get_action_and_value(state)
                        agent.update(
                            state,
                            action,
                            reward,
                            next_state,
                            done or truncated,
                            log_prob,
                            value,
                        )

                    state = next_state
                    episode_reward += reward

                exploration_scheduler.update_exploration(episode_reward)

                scores.append(episode_reward)
                exploration_rates.append(
                    exploration_scheduler.get_exploration_stats()["current_rate"]
                )

                if (episode + 1) % 50 == 0:
                    avg_score = np.mean(scores[-50:])
                    current_rate = exploration_rates[-1]
                    print(
                        f"Episode {episode + 1:4d} | Avg Score: {avg_score:7.2f} | Exploration: {current_rate:.3f}"
                    )

            results[strategy] = {
                "scores": scores,
                "exploration_rates": exploration_rates,
                "episode_rewards": scores,
            }

        self.results["exploration_experiment"] = results
        return results

    def run_hyperparameter_sweep(
        self,
        algorithm: str = "reinforce",
        param_name: str = "lr",
        param_values: List[float] = [0.001, 0.01, 0.1],
        num_episodes: int = 200,
        **kwargs,
    ) -> Dict[str, Dict]:
        """Run hyperparameter sweep

        Args:
            algorithm: Algorithm to test
            param_name: Parameter to sweep
            param_values: Values to test
            num_episodes: Episodes per configuration
            **kwargs: Additional parameters

        Returns:
            Sweep results
        """
        print(f"Running Hyperparameter Sweep for {algorithm}")
        print(f"Parameter: {param_name}, Values: {param_values}")
        print("=" * 50)

        results = {}

        for value in param_values:
            print(f"\\nTesting {param_name} = {value}...")

            params = kwargs.copy()
            params[param_name] = value

            if algorithm == "reinforce":
                agent = REINFORCEAgent(self.state_size, self.action_size, **params)
                training_results = agent.train(
                    self.env.env, num_episodes, print_every=50
                )
            elif algorithm == "actor_critic":
                agent = ActorCriticAgent(self.state_size, self.action_size, **params)
                training_results = agent.train(
                    self.env.env, num_episodes, print_every=50
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            results[f"{param_name}_{value}"] = training_results

        self.results["hyperparameter_sweep"] = results
        return results

    def _log_experiment(self, experiment_name: str, results: Dict[str, Any]):
        """Log experiment results

        Args:
            experiment_name: Name of experiment
            results: Experiment results
        """
        log_entry = {
            "experiment": experiment_name,
            "timestamp": time.time(),
            "environment": self.env_name,
            "final_score": (
                np.mean(results.get("scores", [])[-50:])
                if results.get("scores")
                else None
            ),
            "training_time": results.get("training_time", 0),
            "parameters": {
                k: v
                for k, v in results.items()
                if not isinstance(v, (list, np.ndarray)) and k != "scores"
            },
        }

        self.experiment_log.append(log_entry)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary

        Returns:
            Summary of all experiments
        """
        summary = {
            "environment": self.env_name,
            "total_experiments": len(self.experiment_log),
            "experiments": self.experiment_log,
        }

        if self.results:

            best_score = -float("inf")
            best_experiment = None

            for exp_name, results in self.results.items():
                if isinstance(results, dict) and "scores" in results:
                    final_score = (
                        np.mean(results["scores"][-50:])
                        if len(results["scores"]) >= 50
                        else np.mean(results["scores"])
                    )
                    if final_score > best_score:
                        best_score = final_score
                        best_experiment = exp_name

            summary["best_experiment"] = best_experiment
            summary["best_score"] = best_score

        return summary

    def visualize_results(self, experiment_name: Optional[str] = None):
        """Visualize experiment results

        Args:
            experiment_name: Specific experiment to visualize (optional)
        """
        visualizer = TrainingVisualizer()

        if experiment_name and experiment_name in self.results:
            results = self.results[experiment_name]
            if "scores" in results:
                visualizer.plot_learning_curves(
                    results["scores"], title=f"{experiment_name.upper()} Learning Curve"
                )

            if "policy_losses" in results and "value_losses" in results:
                visualizer.plot_losses(
                    results["policy_losses"],
                    results["value_losses"],
                    title=f"{experiment_name.upper()} Training Losses",
                )

        elif len(self.results) > 1:

            comparison_data = {}
            for exp_name, results in self.results.items():
                if isinstance(results, dict) and "scores" in results:
                    comparison_data[exp_name] = results["scores"]

            if comparison_data:
                visualizer.plot_multiple_curves(
                    comparison_data, title="Algorithm Comparison"
                )

        else:
            print("No results to visualize")


class BenchmarkSuite:
    """Benchmark suite for policy gradient algorithms"""

    def __init__(self):
        """Initialize benchmark suite"""
        self.environments = [
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "Pendulum-v1",
        ]

        self.algorithms = ["reinforce", "actor_critic"]
        self.results = {}

    def run_benchmark(self, episodes_per_env: int = 200) -> Dict[str, Any]:
        """Run comprehensive benchmark

        Args:
            episodes_per_env: Episodes per environment/algorithm combination

        Returns:
            Benchmark results
        """
        print("Running Policy Gradient Benchmark Suite")
        print("=" * 50)

        for env_name in self.environments:
            print(f"\\nBenchmarking {env_name}...")
            self.results[env_name] = {}

            try:
                experiment = PolicyGradientExperiment(env_name)

                for algorithm in self.algorithms:
                    print(f"  Running {algorithm}...")

                    if algorithm == "reinforce":
                        results = experiment.run_reinforce_experiment(episodes_per_env)
                    elif algorithm == "actor_critic":
                        results = experiment.run_actor_critic_experiment(
                            episodes_per_env
                        )

                    self.results[env_name][algorithm] = results

            except Exception as e:
                print(f"  Error benchmarking {env_name}: {e}")
                self.results[env_name] = {"error": str(e)}

        return self.results

    def create_report(self) -> str:
        """Create benchmark report

        Returns:
            Formatted report string
        """
        report = "Policy Gradient Benchmark Report"
        report += "\\n" + "=" * 40 + "\\n\\n"

        for env_name, env_results in self.results.items():
            report += f"Environment: {env_name}\\n"
            report += "-" * 30 + "\\n"

            if "error" in env_results:
                report += f"Error: {env_results['error']}\\n\\n"
                continue

            for alg_name, results in env_results.items():
                if isinstance(results, dict) and "scores" in results:
                    scores = results["scores"]
                    final_avg = (
                        np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
                    )
                    best_score = np.max(scores)
                    training_time = results.get("training_time", 0)

                    report += f"{alg_name.upper()}:\\n"
                    report += f"  Final Average: {final_avg:.2f}\\n"
                    report += f"  Best Score: {best_score:.2f}\\n"
                    report += f"  Training Time: {training_time:.2f}s\\n"

            report += "\\n"

        return report


def run_quick_test(
    env_name: str = "CartPole-v1", algorithm: str = "reinforce", episodes: int = 50
) -> Dict[str, Any]:
    """Run quick test of algorithm

    Args:
        env_name: Environment name
        algorithm: Algorithm to test
        episodes: Number of episodes

    Returns:
        Test results
    """
    print(f"Quick Test: {algorithm} on {env_name} ({episodes} episodes)")

    experiment = PolicyGradientExperiment(env_name)

    if algorithm == "reinforce":
        results = experiment.run_reinforce_experiment(episodes)
    elif algorithm == "actor_critic":
        results = experiment.run_actor_critic_experiment(episodes)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    scores = results.get("scores", [])
    if scores:
        print(f"Final Average Score: {np.mean(scores[-10:]):.2f}")
        print(f"Best Score: {np.max(scores):.2f}")

    return results
