"""
Experiments Module for CA19 Advanced RL Systems

This module provides comprehensive evaluation suites and experiment protocols
for testing and comparing advanced RL algorithms including quantum-enhanced,
neuromorphic, and hybrid systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import sys
import os

# Add current directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import PerformanceTracker, ExperimentManager, MissionConfig
from environments import (
    NeuromorphicEnvironment,
    HybridQuantumClassicalEnvironment,
    MetaLearningEnvironment,
    ContinualLearningEnvironment,
    HierarchicalEnvironment,
)
from hybrid_quantum_classical_rl import HybridQuantumClassicalAgent
from neuromorphic_rl import NeuromorphicActorCritic
from quantum_rl import QuantumEnhancedAgent


class QuantumNeuromorphicComparison:
    """
    Comprehensive comparison between quantum, neuromorphic, and hybrid RL systems
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.results = {}
        self.environments = self._setup_environments()
        self.agents = self._setup_agents()

    def _setup_environments(self) -> Dict[str, Any]:
        """Setup test environments"""
        return {
            "neuromorphic": NeuromorphicEnvironment(
                state_dim=self.config.state_dim, action_dim=self.config.action_dim
            ),
            "hybrid": HybridQuantumClassicalEnvironment(
                state_dim=self.config.state_dim * 2,
                action_dim=self.config.action_dim * 2,
                quantum_complexity=0.8,
            ),
            "meta_learning": MetaLearningEnvironment(
                base_state_dim=self.config.state_dim, num_tasks=5
            ),
            "continual": ContinualLearningEnvironment(
                state_dim=self.config.state_dim, num_phases=3
            ),
            "hierarchical": HierarchicalEnvironment(
                state_dim=self.config.state_dim, num_levels=3
            ),
        }

    def _setup_agents(self) -> Dict[str, Any]:
        """Setup different types of agents"""
        agents = {}

        agents["hybrid_qc"] = HybridQuantumClassicalAgent(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            quantum_dim=self.config.quantum_dim,
            hidden_dim=self.config.hidden_dim,
        )

        agents["neuromorphic"] = NeuromorphicActorCritic(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            neuron_count=self.config.neuron_count,
            synapse_count=self.config.synapse_count,
        )

        agents["quantum_enhanced"] = QuantumEnhancedAgent(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            quantum_dim=self.config.quantum_dim,
        )

        return agents

    def run_comparison_experiment(
        self, num_episodes: int = 100, max_steps: int = 200
    ) -> Dict[str, Any]:
        """Run comprehensive comparison across all agent-environment pairs"""

        print("Starting Quantum-Neuromorphic Comparison Experiment")
        print(f"Episodes: {num_episodes}, Max Steps: {max_steps}")

        results = {}

        for env_name, env in self.environments.items():
            print(f"\nTesting Environment: {env_name}")
            env_results = {}

            for agent_name, agent in self.agents.items():
                print(f"  Training {agent_name} agent...")

                training_rewards, training_lengths = self._train_agent(
                    agent, env, num_episodes, max_steps
                )

                eval_rewards, eval_lengths = self._evaluate_agent(
                    agent, env, num_episodes=20, max_steps=max_steps
                )

                env_results[agent_name] = {
                    "training_rewards": training_rewards,
                    "training_lengths": training_lengths,
                    "eval_rewards": eval_rewards,
                    "eval_lengths": eval_lengths,
                    "final_performance": np.mean(eval_rewards[-10:]),
                    "convergence_episode": self._find_convergence_episode(
                        training_rewards
                    ),
                }

            results[env_name] = env_results

        self.results = results
        return results

    def _train_agent(
        self, agent: Any, env: Any, num_episodes: int, max_steps: int
    ) -> Tuple[List[float], List[int]]:
        """Train an agent in an environment"""
        rewards_history = []
        lengths_history = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < max_steps:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                if hasattr(agent, "train_step"):
                    agent.train_step(state, action, reward, next_state, done)
                elif hasattr(agent, "update"):
                    agent.update(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                episode_length += 1

            rewards_history.append(episode_reward)
            lengths_history.append(episode_length)

            if (episode + 1) % 20 == 0:
                print(
                    f"    Episode {episode + 1}/{num_episodes}, "
                    f"Avg Reward: {np.mean(rewards_history[-20:]):.2f}"
                )

        return rewards_history, lengths_history

    def _evaluate_agent(
        self, agent: Any, env: Any, num_episodes: int = 20, max_steps: int = 200
    ) -> Tuple[List[float], List[int]]:
        """Evaluate trained agent"""
        rewards_history = []
        lengths_history = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < max_steps:
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward
                episode_length += 1

            rewards_history.append(episode_reward)
            lengths_history.append(episode_length)

        return rewards_history, lengths_history

    def _find_convergence_episode(
        self, rewards: List[float], window_size: int = 20
    ) -> Optional[int]:
        """Find episode where agent converged (stable performance)"""
        if len(rewards) < window_size * 2:
            return None

        for i in range(window_size, len(rewards) - window_size):
            early_window = rewards[i - window_size : i]
            late_window = rewards[i : i + window_size]

            early_std = np.std(early_window)
            late_std = np.std(late_window)

            if late_std < early_std * 0.5 and np.mean(late_window) > np.mean(
                early_window
            ):
                return i

        return None

    def plot_comparison_results(self, save_path: Optional[str] = None):
        """Plot comprehensive comparison results"""
        if not self.results:
            print("No results to plot. Run experiment first.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Quantum-Neuromorphic RL Systems Comparison", fontsize=16)

        environments = list(self.results.keys())
        agents = list(self.results[environments[0]].keys())

        for i, env_name in enumerate(environments):
            ax = axes[i // 3, i % 3]
            env_data = self.results[env_name]

            for agent_name in agents:
                if agent_name in env_data:
                    rewards = env_data[agent_name]["training_rewards"]
                    ax.plot(rewards, label=agent_name, alpha=0.7)

            ax.set_title(f'{env_name.replace("_", " ").title()}')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def generate_performance_report(self) -> str:
        """Generate detailed performance report"""
        if not self.results:
            return "No results available. Run experiment first."

        report = []
        report.append("# Quantum-Neuromorphic RL Comparison Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for env_name, env_results in self.results.items():
            report.append(f"## Environment: {env_name.replace('_', ' ').title()}")
            report.append("")

            table_data = []
            for agent_name, agent_data in env_results.items():
                table_data.append(
                    {
                        "Agent": agent_name,
                        "Final Performance": ".2f",
                        "Convergence Episode": agent_data.get(
                            "convergence_episode", "N/A"
                        ),
                        "Training Episodes": len(agent_data["training_rewards"]),
                    }
                )

            df = pd.DataFrame(table_data)
            report.append(df.to_markdown(index=False))
            report.append("")

            best_agent = max(
                env_results.keys(), key=lambda x: env_results[x]["final_performance"]
            )
            report.append(f"**Best Performing Agent:** {best_agent}")
            report.append(".2f")
            report.append("")

        return "\n".join(report)


class AblationStudy:
    """
    Ablation studies to understand component contributions
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.ablation_results = {}

    def run_quantum_ablation(self, env: Any, num_episodes: int = 50) -> Dict[str, Any]:
        """Ablation study for quantum components"""
        print("Running Quantum Component Ablation Study")

        variants = {
            "full_quantum": {"use_quantum": True, "use_classical": True},
            "quantum_only": {"use_quantum": True, "use_classical": False},
            "classical_only": {"use_quantum": False, "use_classical": True},
            "no_quantum": {"use_quantum": False, "use_classical": False},
        }

        results = {}

        for variant_name, params in variants.items():
            print(f"  Testing variant: {variant_name}")

            agent = HybridQuantumClassicalAgent(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                quantum_dim=self.config.quantum_dim if params["use_quantum"] else 0,
                hidden_dim=self.config.hidden_dim,
            )

            if not params["use_quantum"]:
                agent.quantum_circuit = None

            training_rewards, _ = self._train_agent_simple(agent, env, num_episodes)
            eval_reward = self._evaluate_agent_simple(agent, env)

            results[variant_name] = {
                "training_rewards": training_rewards,
                "final_performance": eval_reward,
                "parameters": params,
            }

        self.ablation_results["quantum"] = results
        return results

    def run_neuromorphic_ablation(
        self, env: Any, num_episodes: int = 50
    ) -> Dict[str, Any]:
        """Ablation study for neuromorphic components"""
        print("Running Neuromorphic Component Ablation Study")

        variants = {
            "full_neuromorphic": {
                "use_stdp": True,
                "use_spiking": True,
                "use_dopamine": True,
            },
            "no_stdp": {"use_stdp": False, "use_spiking": True, "use_dopamine": True},
            "no_spiking": {
                "use_stdp": True,
                "use_spiking": False,
                "use_dopamine": True,
            },
            "no_dopamine": {
                "use_stdp": True,
                "use_spiking": True,
                "use_dopamine": False,
            },
        }

        results = {}

        for variant_name, params in variants.items():
            print(f"  Testing variant: {variant_name}")

            agent = NeuromorphicActorCritic(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                neuron_count=self.config.neuron_count,
                synapse_count=self.config.synapse_count,
            )

            if not params["use_stdp"]:
                for synapse in agent.network.synapses:
                    synapse.stdp_enabled = False

            if not params["use_spiking"]:
                for neuron in agent.network.neurons:
                    neuron.spiking_enabled = False

            if not params["use_dopamine"]:
                agent.dopamine_modulation = 0.0

            training_rewards, _ = self._train_agent_simple(agent, env, num_episodes)
            eval_reward = self._evaluate_agent_simple(agent, env)

            results[variant_name] = {
                "training_rewards": training_rewards,
                "final_performance": eval_reward,
                "parameters": params,
            }

        self.ablation_results["neuromorphic"] = results
        return results

    def _train_agent_simple(
        self, agent: Any, env: Any, num_episodes: int
    ) -> Tuple[List[float], List[int]]:
        """Simplified training for ablation studies"""
        rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 100:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                if hasattr(agent, "train_step"):
                    agent.train_step(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                step_count += 1

            rewards.append(episode_reward)

        return rewards, []

    def _evaluate_agent_simple(self, agent: Any, env: Any) -> float:
        """Simplified evaluation"""
        total_reward = 0

        for episode in range(10):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 100:
                action = agent.select_action(state, training=False)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                episode_reward += reward
                step_count += 1

            total_reward += episode_reward

        return total_reward / 10


class ScalabilityAnalysis:
    """
    Analyze scalability of different RL approaches
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.scalability_results = {}

    def run_scalability_test(
        self, problem_sizes: List[int] = [4, 8, 16, 32]
    ) -> Dict[str, Any]:
        """Test scalability across different problem sizes"""
        print("Running Scalability Analysis")

        results = {}

        for size in problem_sizes:
            print(f"  Testing problem size: {size}")

            env = HybridQuantumClassicalEnvironment(
                state_dim=size, action_dim=min(size * 2, 16), quantum_complexity=0.8
            )

            agents = {
                "hybrid_qc": HybridQuantumClassicalAgent(
                    state_dim=size,
                    action_dim=min(size * 2, 16),
                    quantum_dim=min(size, 8),
                    hidden_dim=size,
                ),
                "neuromorphic": NeuromorphicActorCritic(
                    state_dim=size,
                    action_dim=min(size * 2, 16),
                    neuron_count=size * 2,
                    synapse_count=size * 4,
                ),
                "quantum_enhanced": QuantumEnhancedAgent(
                    state_dim=size,
                    action_dim=min(size * 2, 16),
                    quantum_dim=min(size, 8),
                ),
            }

            size_results = {}

            for agent_name, agent in agents.items():
                try:
                    training_rewards, _ = self._quick_train(agent, env, num_episodes=20)
                    eval_reward = self._quick_evaluate(agent, env)

                    size_results[agent_name] = {
                        "training_rewards": training_rewards,
                        "final_performance": eval_reward,
                        "training_time": len(training_rewards) * 0.1,  # Estimated
                        "success": True,
                    }

                except Exception as e:
                    size_results[agent_name] = {"error": str(e), "success": False}

            results[size] = size_results

        self.scalability_results = results
        return results

    def _quick_train(
        self, agent: Any, env: Any, num_episodes: int
    ) -> Tuple[List[float], List[int]]:
        """Very quick training for scalability testing"""
        rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 50:  # Shorter episodes
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                if hasattr(agent, "train_step"):
                    agent.train_step(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                step_count += 1

            rewards.append(episode_reward)

        return rewards, []

    def _quick_evaluate(self, agent: Any, env: Any) -> float:
        """Quick evaluation"""
        total_reward = 0

        for episode in range(5):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 50:
                action = agent.select_action(state, training=False)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                episode_reward += reward
                step_count += 1

            total_reward += episode_reward

        return total_reward / 5

    def plot_scalability_results(self, save_path: Optional[str] = None):
        """Plot scalability analysis results"""
        if not self.scalability_results:
            print("No scalability results to plot. Run analysis first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Scalability Analysis: Performance vs Problem Size", fontsize=14)

        sizes = list(self.scalability_results.keys())
        agents = ["hybrid_qc", "neuromorphic", "quantum_enhanced"]

        ax1 = axes[0]
        for agent in agents:
            performances = []
            for size in sizes:
                if (
                    agent in self.scalability_results[size]
                    and self.scalability_results[size][agent]["success"]
                ):
                    performances.append(
                        self.scalability_results[size][agent]["final_performance"]
                    )
                else:
                    performances.append(0)

            ax1.plot(sizes, performances, "o-", label=agent, markersize=8)

        ax1.set_xlabel("Problem Size (State Dimension)")
        ax1.set_ylabel("Final Performance (Reward)")
        ax1.set_title("Performance Scaling")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log", base=2)

        ax2 = axes[1]
        for agent in agents:
            success_rates = []
            for size in sizes:
                success = (
                    1
                    if (
                        agent in self.scalability_results[size]
                        and self.scalability_results[size][agent]["success"]
                    )
                    else 0
                )
                success_rates.append(success)

            ax2.plot(sizes, success_rates, "s-", label=agent, markersize=8)

        ax2.set_xlabel("Problem Size (State Dimension)")
        ax2.set_ylabel("Success Rate")
        ax2.set_title("Training Success Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log", base=2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


class ExperimentRunner:
    """
    High-level experiment runner for coordinating multiple studies
    """

    def __init__(self, config: MissionConfig, save_dir: str = "experiment_results"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.experiments = {
            "comparison": QuantumNeuromorphicComparison(config),
            "ablation": AblationStudy(config),
            "scalability": ScalabilityAnalysis(config),
        }

    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all experiments"""
        print("Starting Complete Experiment Suite")
        print(f"Results will be saved to: {self.save_dir}")

        all_results = {}

        print("\n" + "=" * 50)
        print("PHASE 1: Quantum-Neuromorphic Comparison")
        print("=" * 50)

        comparison_results = self.experiments["comparison"].run_comparison_experiment(
            num_episodes=self.config.num_episodes, max_steps=self.config.max_steps
        )
        all_results["comparison"] = comparison_results

        print("\n" + "=" * 50)
        print("PHASE 2: Ablation Studies")
        print("=" * 50)

        ablation_results = {}
        env = NeuromorphicEnvironment()  # Use simple environment for ablation

        ablation_results["quantum"] = self.experiments["ablation"].run_quantum_ablation(
            env, num_episodes=30
        )
        ablation_results["neuromorphic"] = self.experiments[
            "ablation"
        ].run_neuromorphic_ablation(env, num_episodes=30)
        all_results["ablation"] = ablation_results

        print("\n" + "=" * 50)
        print("PHASE 3: Scalability Analysis")
        print("=" * 50)

        scalability_results = self.experiments["scalability"].run_scalability_test()
        all_results["scalability"] = scalability_results

        self._save_results(all_results)

        self._generate_reports(all_results)

        print("\n" + "=" * 50)
        print("EXPERIMENT SUITE COMPLETED")
        print("=" * 50)

        return all_results

    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = self.save_dir / f"experiment_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {json_path}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)  # Convert other objects to string

    def _generate_reports(self, results: Dict[str, Any]):
        """Generate comprehensive reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_path = self.save_dir / f"performance_report_{timestamp}.md"
        with open(report_path, "w") as f:
            f.write(self.experiments["comparison"].generate_performance_report())

        try:
            self.experiments["comparison"].plot_comparison_results(
                save_path=str(self.save_dir / f"comparison_plot_{timestamp}.png")
            )
            self.experiments["scalability"].plot_scalability_results(
                save_path=str(self.save_dir / f"scalability_plot_{timestamp}.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

        print(f"Reports generated in: {self.save_dir}")


def run_quick_comparison(config: Optional[MissionConfig] = None) -> Dict[str, Any]:
    """Run a quick comparison experiment"""
    if config is None:
        config = MissionConfig()

    experiment = QuantumNeuromorphicComparison(config)
    results = experiment.run_comparison_experiment(num_episodes=50, max_steps=100)

    print("\nQuick Comparison Results:")
    print(experiment.generate_performance_report())

    return results


def run_ablation_study(config: Optional[MissionConfig] = None) -> Dict[str, Any]:
    """Run ablation studies"""
    if config is None:
        config = MissionConfig()

    study = AblationStudy(config)
    env = NeuromorphicEnvironment()

    results = {
        "quantum": study.run_quantum_ablation(env, num_episodes=30),
        "neuromorphic": study.run_neuromorphic_ablation(env, num_episodes=30),
    }

    return results


def benchmark_scalability(config: Optional[MissionConfig] = None) -> Dict[str, Any]:
    """Run scalability benchmark"""
    if config is None:
        config = MissionConfig()

    analysis = ScalabilityAnalysis(config)
    results = analysis.run_scalability_test(problem_sizes=[4, 8, 16])

    return results
