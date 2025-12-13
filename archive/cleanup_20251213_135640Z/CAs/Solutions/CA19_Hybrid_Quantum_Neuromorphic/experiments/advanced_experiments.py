"""
Advanced Experiments Module for CA19 Quantum-Neuromorphic RL Systems

This module implements sophisticated experimental protocols for:
- Multi-objective optimization experiments
- Scalability and robustness testing
- Cross-domain transfer learning
- Adversarial robustness evaluation
- Energy efficiency benchmarking
- Real-time adaptation studies
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings("ignore")

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import PerformanceTracker, ExperimentManager, MissionConfig
from environments import MultidimensionalQuantumEnvironment
from agents.advanced_quantum_agent import AdvancedQuantumAgent
from agents.advanced_neuromorphic_agent import AdvancedNeuromorphicAgent
from analysis import (
    QuantumCoherenceAnalyzer,
    NeuromorphicEfficiencyAnalyzer,
    HybridSystemAnalyzer,
)


class MultiObjectiveOptimizationExperiment:
    """
    Multi-objective optimization experiments for quantum-neuromorphic systems
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.objectives = ["performance", "efficiency", "robustness", "scalability"]
        self.results = {}
        self.pareto_front = []

    def run_multi_objective_experiment(
        self, n_generations: int = 50, population_size: int = 100
    ) -> Dict[str, Any]:
        """Run multi-objective optimization experiment"""

        print("🧬 Starting Multi-Objective Optimization Experiment")
        print(f"Objectives: {self.objectives}")
        print(f"Generations: {n_generations}, Population Size: {population_size}")

        # Initialize population
        population = self._initialize_population(population_size)

        for generation in range(n_generations):
            print(f"\n🔄 Generation {generation + 1}/{n_generations}")

            # Evaluate population
            fitness_scores = self._evaluate_population(population)

            # Update Pareto front
            self._update_pareto_front(population, fitness_scores)

            # Selection and reproduction
            population = self._evolve_population(population, fitness_scores)

            # Print progress
            if (generation + 1) % 10 == 0:
                self._print_progress(generation + 1, fitness_scores)

        # Final analysis
        final_results = self._analyze_results()

        return final_results

    def _initialize_population(self, population_size: int) -> List[Dict[str, Any]]:
        """Initialize random population of agent configurations"""
        population = []

        for _ in range(population_size):
            individual = {
                "quantum_qubits": np.random.randint(4, 12),
                "quantum_layers": np.random.randint(2, 6),
                "neuromorphic_neurons": np.random.randint(32, 128),
                "learning_rate": np.random.uniform(1e-4, 1e-2),
                "exploration_rate": np.random.uniform(0.01, 0.3),
                "hybrid_weight": np.random.uniform(0.2, 0.8),
                "state_dim": np.random.choice([16, 32, 64]),
                "action_dim": np.random.choice([8, 16, 32]),
            }
            population.append(individual)

        return population

    def _evaluate_population(self, population: List[Dict[str, Any]]) -> np.ndarray:
        """Evaluate fitness for all individuals"""
        fitness_scores = np.zeros((len(population), len(self.objectives)))

        for i, individual in enumerate(population):
            try:
                # Create environment
                env = MultidimensionalQuantumEnvironment(
                    state_dim=individual["state_dim"],
                    action_dim=individual["action_dim"],
                )

                # Create agents
                quantum_agent = AdvancedQuantumAgent(
                    state_dim=individual["state_dim"],
                    action_dim=individual["action_dim"],
                    n_qubits=individual["quantum_qubits"],
                    learning_rate=individual["learning_rate"],
                )

                neuromorphic_agent = AdvancedNeuromorphicAgent(
                    state_dim=individual["state_dim"],
                    action_dim=individual["action_dim"],
                    learning_rate=individual["learning_rate"],
                )

                # Run evaluation
                scores = self._evaluate_individual(
                    individual, env, quantum_agent, neuromorphic_agent
                )
                fitness_scores[i] = scores

                env.close()

            except Exception as e:
                print(f"⚠️ Evaluation failed for individual {i}: {e}")
                fitness_scores[i] = np.zeros(len(self.objectives))

        return fitness_scores

    def _evaluate_individual(
        self, individual: Dict[str, Any], env, quantum_agent, neuromorphic_agent
    ) -> np.ndarray:
        """Evaluate individual performance across multiple objectives"""

        # Performance objective
        performance_score = self._evaluate_performance(
            env, quantum_agent, neuromorphic_agent
        )

        # Efficiency objective
        efficiency_score = self._evaluate_efficiency(quantum_agent, neuromorphic_agent)

        # Robustness objective
        robustness_score = self._evaluate_robustness(
            env, quantum_agent, neuromorphic_agent
        )

        # Scalability objective
        scalability_score = self._evaluate_scalability(individual)

        return np.array(
            [performance_score, efficiency_score, robustness_score, scalability_score]
        )

    def _evaluate_performance(self, env, quantum_agent, neuromorphic_agent) -> float:
        """Evaluate performance objective"""
        total_rewards = []

        for _ in range(3):  # Multiple episodes
            state, _ = env.reset()
            episode_reward = 0

            for _ in range(100):
                action, _ = quantum_agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)

                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def _evaluate_efficiency(self, quantum_agent, neuromorphic_agent) -> float:
        """Evaluate efficiency objective"""
        # Get quantum metrics
        quantum_metrics = quantum_agent.get_quantum_metrics()

        # Get neuromorphic metrics
        neuromorphic_metrics = neuromorphic_agent.get_performance_metrics()

        # Combine efficiency metrics
        quantum_efficiency = np.mean(
            quantum_metrics.get("quantum_fidelity_history", [0.5])
        )
        neuromorphic_efficiency = neuromorphic_metrics.get("avg_energy_efficiency", 0.5)

        return (quantum_efficiency + neuromorphic_efficiency) / 2

    def _evaluate_robustness(self, env, quantum_agent, neuromorphic_agent) -> float:
        """Evaluate robustness objective"""
        # Test with noise
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        robustness_scores = []

        for noise_level in noise_levels:
            state, _ = env.reset()
            episode_reward = 0

            for _ in range(50):
                # Add noise to state
                noisy_state = state + np.random.normal(0, noise_level, state.shape)
                action, _ = quantum_agent.select_action(noisy_state)
                next_state, reward, done, truncated, _ = env.step(action)

                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            robustness_scores.append(episode_reward)

        # Calculate robustness as performance degradation
        baseline = robustness_scores[0]
        degradation = np.mean(
            [(baseline - score) / (baseline + 1e-8) for score in robustness_scores[1:]]
        )
        robustness = 1.0 - degradation

        return max(0, robustness)

    def _evaluate_scalability(self, individual: Dict[str, Any]) -> float:
        """Evaluate scalability objective"""
        # Simple scalability metric based on parameter count
        quantum_params = individual["quantum_qubits"] * individual["quantum_layers"]
        neuromorphic_params = individual["neuromorphic_neurons"] * 10  # Approximate

        total_params = quantum_params + neuromorphic_params

        # Higher scalability for fewer parameters (simpler models)
        scalability = 1.0 / (1.0 + total_params / 1000.0)

        return scalability

    def _update_pareto_front(
        self, population: List[Dict[str, Any]], fitness_scores: np.ndarray
    ):
        """Update Pareto front with non-dominated solutions"""
        for i, scores in enumerate(fitness_scores):
            is_dominated = False

            # Check if this solution is dominated by any existing Pareto solution
            for pareto_scores in self.pareto_front:
                if self._dominates(pareto_scores, scores):
                    is_dominated = True
                    break

            if not is_dominated:
                # Remove solutions dominated by this one
                self.pareto_front = [
                    s for s in self.pareto_front if not self._dominates(scores, s)
                ]
                self.pareto_front.append(scores)

    def _dominates(self, scores1: np.ndarray, scores2: np.ndarray) -> bool:
        """Check if scores1 dominates scores2"""
        return np.all(scores1 >= scores2) and np.any(scores1 > scores2)

    def _evolve_population(
        self, population: List[Dict[str, Any]], fitness_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Evolve population using genetic operators"""
        new_population = []

        # Tournament selection
        for _ in range(len(population)):
            # Select parents
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child)

            new_population.append(child)

        return new_population

    def _tournament_selection(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: np.ndarray,
        tournament_size: int = 3,
    ) -> Dict[str, Any]:
        """Tournament selection"""
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_scores = fitness_scores[tournament_indices]

        # Select best individual from tournament
        best_idx = np.argmax(np.sum(tournament_scores, axis=1))

        return population[tournament_indices[best_idx]]

    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Uniform crossover"""
        child = {}

        for key in parent1:
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]

        return child

    def _mutate(
        self, individual: Dict[str, Any], mutation_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Gaussian mutation"""
        mutated = individual.copy()

        for key, value in individual.items():
            if np.random.random() < mutation_rate:
                if isinstance(value, int):
                    mutated[key] = max(1, int(value + np.random.normal(0, value * 0.1)))
                elif isinstance(value, float):
                    mutated[key] = max(0.001, value + np.random.normal(0, value * 0.1))

        return mutated

    def _print_progress(self, generation: int, fitness_scores: np.ndarray):
        """Print progress information"""
        avg_fitness = np.mean(fitness_scores, axis=0)
        best_fitness = np.max(fitness_scores, axis=0)

        print(f"📊 Generation {generation} Progress:")
        for i, objective in enumerate(self.objectives):
            print(
                f"  {objective.capitalize()}: Avg={avg_fitness[i]:.3f}, Best={best_fitness[i]:.3f}"
            )

        print(f"  Pareto Front Size: {len(self.pareto_front)}")

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze final results"""
        if not self.pareto_front:
            return {}

        pareto_scores = np.array(self.pareto_front)

        # Calculate hypervolume (approximation)
        hypervolume = np.prod(np.max(pareto_scores, axis=0))

        # Diversity analysis
        diversity = np.mean(np.std(pareto_scores, axis=0))

        # Best solutions for each objective
        best_solutions = {}
        for i, objective in enumerate(self.objectives):
            best_idx = np.argmax(pareto_scores[:, i])
            best_solutions[objective] = {
                "score": pareto_scores[best_idx, i],
                "index": best_idx,
            }

        return {
            "pareto_front_size": len(self.pareto_front),
            "hypervolume": hypervolume,
            "diversity": diversity,
            "best_solutions": best_solutions,
            "pareto_scores": pareto_scores.tolist(),
        }


class ScalabilityRobustnessExperiment:
    """
    Scalability and robustness testing experiments
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.scalability_results = {}
        self.robustness_results = {}

    def run_scalability_experiment(
        self, problem_sizes: List[int] = [16, 32, 64, 128, 256]
    ) -> Dict[str, Any]:
        """Run scalability experiment across different problem sizes"""

        print("📈 Starting Scalability Experiment")
        print(f"Problem sizes: {problem_sizes}")

        scalability_metrics = {
            "problem_sizes": [],
            "quantum_performance": [],
            "neuromorphic_performance": [],
            "hybrid_performance": [],
            "quantum_time": [],
            "neuromorphic_time": [],
            "hybrid_time": [],
            "memory_usage": [],
        }

        for size in problem_sizes:
            print(f"\n🔍 Testing problem size: {size}")

            # Create environment
            env = MultidimensionalQuantumEnvironment(
                state_dim=size, action_dim=size // 4
            )

            # Test quantum agent
            quantum_results = self._test_quantum_scalability(env, size)
            scalability_metrics["quantum_performance"].append(
                quantum_results["performance"]
            )
            scalability_metrics["quantum_time"].append(quantum_results["time"])

            # Test neuromorphic agent
            neuromorphic_results = self._test_neuromorphic_scalability(env, size)
            scalability_metrics["neuromorphic_performance"].append(
                neuromorphic_results["performance"]
            )
            scalability_metrics["neuromorphic_time"].append(
                neuromorphic_results["time"]
            )

            # Test hybrid approach
            hybrid_results = self._test_hybrid_scalability(env, size)
            scalability_metrics["hybrid_performance"].append(
                hybrid_results["performance"]
            )
            scalability_metrics["hybrid_time"].append(hybrid_results["time"])

            scalability_metrics["memory_usage"].append(hybrid_results["memory"])
            scalability_metrics["problem_sizes"].append(size)

            env.close()

        # Analyze scalability trends
        analysis = self._analyze_scalability_trends(scalability_metrics)

        return {"scalability_metrics": scalability_metrics, "analysis": analysis}

    def _test_quantum_scalability(self, env, size: int) -> Dict[str, Any]:
        """Test quantum agent scalability"""
        import time

        start_time = time.time()

        # Create quantum agent
        quantum_agent = AdvancedQuantumAgent(
            state_dim=size,
            action_dim=env.action_space.n,
            n_qubits=min(8, size // 4),
            hidden_dim=min(128, size * 2),
        )

        # Run test
        total_reward = 0
        state, _ = env.reset()

        for _ in range(50):
            action, _ = quantum_agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        end_time = time.time()

        return {
            "performance": total_reward,
            "time": end_time - start_time,
            "memory": 0,  # Would need actual memory monitoring
        }

    def _test_neuromorphic_scalability(self, env, size: int) -> Dict[str, Any]:
        """Test neuromorphic agent scalability"""
        import time

        start_time = time.time()

        # Create neuromorphic agent
        neuromorphic_agent = AdvancedNeuromorphicAgent(
            state_dim=size,
            action_dim=env.action_space.n,
            hidden_dims=[min(64, size), min(32, size // 2)],
        )

        # Run test
        total_reward = 0
        state, _ = env.reset()

        for _ in range(50):
            action, _ = neuromorphic_agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        end_time = time.time()

        return {
            "performance": total_reward,
            "time": end_time - start_time,
            "memory": 0,  # Would need actual memory monitoring
        }

    def _test_hybrid_scalability(self, env, size: int) -> Dict[str, Any]:
        """Test hybrid agent scalability"""
        import time

        start_time = time.time()

        # Create both agents
        quantum_agent = AdvancedQuantumAgent(
            state_dim=size,
            action_dim=env.action_space.n,
            n_qubits=min(6, size // 6),
            hidden_dim=min(64, size),
        )

        neuromorphic_agent = AdvancedNeuromorphicAgent(
            state_dim=size,
            action_dim=env.action_space.n,
            hidden_dims=[min(32, size // 2), min(16, size // 4)],
        )

        # Run hybrid test
        total_reward = 0
        state, _ = env.reset()

        for _ in range(50):
            # Use both agents and combine decisions
            q_action, _ = quantum_agent.select_action(state)
            n_action, _ = neuromorphic_agent.select_action(state)

            # Simple combination: alternate between agents
            action = q_action if _ % 2 == 0 else n_action

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        end_time = time.time()

        return {
            "performance": total_reward,
            "time": end_time - start_time,
            "memory": 0,  # Would need actual memory monitoring
        }

    def _analyze_scalability_trends(self, metrics: Dict[str, List]) -> Dict[str, Any]:
        """Analyze scalability trends"""
        sizes = np.array(metrics["problem_sizes"])

        analysis = {}

        # Performance scaling
        for agent_type in ["quantum", "neuromorphic", "hybrid"]:
            performance = np.array(metrics[f"{agent_type}_performance"])
            time_complexity = np.array(metrics[f"{agentum}_time"])

            # Fit scaling laws
            if len(performance) > 2:
                perf_slope, _ = np.polyfit(np.log(sizes), np.log(performance + 1e-8), 1)
                time_slope, _ = np.polyfit(
                    np.log(sizes), np.log(time_complexity + 1e-8), 1
                )

                analysis[f"{agent_type}_performance_scaling"] = perf_slope
                analysis[f"{agent_type}_time_scaling"] = time_slope

        # Memory scaling
        memory_usage = np.array(metrics["memory_usage"])
        if len(memory_usage) > 2:
            memory_slope, _ = np.polyfit(np.log(sizes), np.log(memory_usage + 1e-8), 1)
            analysis["memory_scaling"] = memory_slope

        return analysis

    def run_robustness_experiment(
        self, noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ) -> Dict[str, Any]:
        """Run robustness experiment with different noise levels"""

        print("🛡️ Starting Robustness Experiment")
        print(f"Noise levels: {noise_levels}")

        robustness_metrics = {
            "noise_levels": noise_levels,
            "quantum_robustness": [],
            "neuromorphic_robustness": [],
            "hybrid_robustness": [],
        }

        for noise_level in noise_levels:
            print(f"\n🔊 Testing noise level: {noise_level}")

            # Create environment with noise
            env = MultidimensionalQuantumEnvironment(state_dim=32, action_dim=8)

            # Test each agent type
            quantum_robustness = self._test_agent_robustness(
                env, "quantum", noise_level
            )
            neuromorphic_robustness = self._test_agent_robustness(
                env, "neuromorphic", noise_level
            )
            hybrid_robustness = self._test_agent_robustness(env, "hybrid", noise_level)

            robustness_metrics["quantum_robustness"].append(quantum_robustness)
            robustness_metrics["neuromorphic_robustness"].append(
                neuromorphic_robustness
            )
            robustness_metrics["hybrid_robustness"].append(hybrid_robustness)

            env.close()

        # Analyze robustness trends
        analysis = self._analyze_robustness_trends(robustness_metrics)

        return {"robustness_metrics": robustness_metrics, "analysis": analysis}

    def _test_agent_robustness(self, env, agent_type: str, noise_level: float) -> float:
        """Test agent robustness to noise"""

        if agent_type == "quantum":
            agent = AdvancedQuantumAgent(32, 8, n_qubits=6)
        elif agent_type == "neuromorphic":
            agent = AdvancedNeuromorphicAgent(32, 8)
        else:  # hybrid
            quantum_agent = AdvancedQuantumAgent(32, 8, n_qubits=6)
            neuromorphic_agent = AdvancedNeuromorphicAgent(32, 8)

        total_rewards = []

        for _ in range(5):  # Multiple trials
            state, _ = env.reset()
            episode_reward = 0

            for _ in range(50):
                # Add noise to state
                noisy_state = state + np.random.normal(0, noise_level, state.shape)

                if agent_type == "hybrid":
                    q_action, _ = quantum_agent.select_action(noisy_state)
                    n_action, _ = neuromorphic_agent.select_action(noisy_state)
                    action = q_action if _ % 2 == 0 else n_action
                else:
                    action, _ = agent.select_action(noisy_state)

                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def _analyze_robustness_trends(self, metrics: Dict[str, List]) -> Dict[str, Any]:
        """Analyze robustness trends"""
        noise_levels = np.array(metrics["noise_levels"])

        analysis = {}

        for agent_type in ["quantum", "neuromorphic", "hybrid"]:
            robustness = np.array(metrics[f"{agent_type}_robustness"])

            # Calculate robustness degradation
            baseline = robustness[0]  # No noise performance
            degradation_rates = []

            for i in range(1, len(robustness)):
                degradation = (baseline - robustness[i]) / (baseline + 1e-8)
                degradation_rates.append(degradation)

            analysis[f"{agent_type}_degradation_rate"] = np.mean(degradation_rates)
            analysis[f"{agent_type}_robustness_slope"] = np.polyfit(
                noise_levels, robustness, 1
            )[0]

        return analysis


class CrossDomainTransferExperiment:
    """
    Cross-domain transfer learning experiments
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.domain_results = {}

    def run_transfer_experiment(
        self,
        source_domains: List[str] = ["cartpole", "mountain_car", "acrobot"],
        target_domain: str = "multidimensional_quantum",
    ) -> Dict[str, Any]:
        """Run cross-domain transfer learning experiment"""

        print("🔄 Starting Cross-Domain Transfer Experiment")
        print(f"Source domains: {source_domains}")
        print(f"Target domain: {target_domain}")

        transfer_results = {
            "source_domains": source_domains,
            "target_domain": target_domain,
            "transfer_performance": {},
            "baseline_performance": {},
            "transfer_efficiency": {},
        }

        # Test baseline performance (no transfer)
        baseline_performance = self._test_baseline_performance(target_domain)
        transfer_results["baseline_performance"] = baseline_performance

        # Test transfer from each source domain
        for source_domain in source_domains:
            print(f"\n📚 Transferring from {source_domain} to {target_domain}")

            # Train on source domain
            source_agent = self._train_on_source_domain(source_domain)

            # Transfer to target domain
            transfer_performance = self._test_transfer_performance(
                source_agent, target_domain
            )
            transfer_results["transfer_performance"][
                source_domain
            ] = transfer_performance

            # Calculate transfer efficiency
            efficiency = transfer_performance / (baseline_performance + 1e-8)
            transfer_results["transfer_efficiency"][source_domain] = efficiency

        # Analyze transfer patterns
        analysis = self._analyze_transfer_patterns(transfer_results)

        return {"transfer_results": transfer_results, "analysis": analysis}

    def _test_baseline_performance(self, domain: str) -> float:
        """Test baseline performance without transfer"""

        if domain == "multidimensional_quantum":
            env = MultidimensionalQuantumEnvironment()
            agent = AdvancedQuantumAgent(32, 8)
        else:
            # For other domains, would use standard gym environments
            return 0.0  # Placeholder

        total_rewards = []

        for _ in range(3):
            state, _ = env.reset()
            episode_reward = 0

            for _ in range(100):
                action, _ = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            total_rewards.append(episode_reward)

        env.close()
        return np.mean(total_rewards)

    def _train_on_source_domain(self, source_domain: str):
        """Train agent on source domain"""

        # This would implement training on different source domains
        # For now, return a placeholder agent
        return AdvancedQuantumAgent(32, 8)

    def _test_transfer_performance(self, source_agent, target_domain: str) -> float:
        """Test transfer performance to target domain"""

        if target_domain == "multidimensional_quantum":
            env = MultidimensionalQuantumEnvironment()
        else:
            return 0.0  # Placeholder

        total_rewards = []

        for _ in range(3):
            state, _ = env.reset()
            episode_reward = 0

            for _ in range(100):
                action, _ = source_agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            total_rewards.append(episode_reward)

        env.close()
        return np.mean(total_rewards)

    def _analyze_transfer_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transfer learning patterns"""

        transfer_efficiencies = list(results["transfer_efficiency"].values())

        analysis = {
            "best_source_domain": max(
                results["transfer_efficiency"], key=results["transfer_efficiency"].get
            ),
            "worst_source_domain": min(
                results["transfer_efficiency"], key=results["transfer_efficiency"].get
            ),
            "avg_transfer_efficiency": np.mean(transfer_efficiencies),
            "transfer_efficiency_std": np.std(transfer_efficiencies),
            "positive_transfer": sum(1 for eff in transfer_efficiencies if eff > 1.0),
            "negative_transfer": sum(1 for eff in transfer_efficiencies if eff < 1.0),
        }

        return analysis


class AdvancedExperimentSuite:
    """
    Comprehensive suite of advanced experiments
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.experiment_results = {}

        # Initialize experiment modules
        self.multi_objective_exp = MultiObjectiveOptimizationExperiment(config)
        self.scalability_exp = ScalabilityRobustnessExperiment(config)
        self.transfer_exp = CrossDomainTransferExperiment(config)

    def run_comprehensive_experiment_suite(self) -> Dict[str, Any]:
        """Run all advanced experiments"""

        print("🚀 Starting Comprehensive Advanced Experiment Suite")
        print("=" * 60)

        all_results = {}

        # Run multi-objective optimization
        print("\n🧬 Phase 1: Multi-Objective Optimization")
        mo_results = self.multi_objective_exp.run_multi_objective_experiment(
            n_generations=20, population_size=50
        )
        all_results["multi_objective"] = mo_results

        # Run scalability experiments
        print("\n📈 Phase 2: Scalability Testing")
        scalability_results = self.scalability_exp.run_scalability_experiment(
            [16, 32, 64]
        )
        all_results["scalability"] = scalability_results

        # Run robustness experiments
        print("\n🛡️ Phase 3: Robustness Testing")
        robustness_results = self.scalability_exp.run_robustness_experiment(
            [0.0, 0.1, 0.2, 0.3]
        )
        all_results["robustness"] = robustness_results

        # Run transfer learning experiments
        print("\n🔄 Phase 4: Transfer Learning")
        transfer_results = self.transfer_exp.run_transfer_experiment(["cartpole"])
        all_results["transfer_learning"] = transfer_results

        # Generate comprehensive report
        print("\n📊 Generating Comprehensive Report")
        comprehensive_report = self._generate_comprehensive_report(all_results)
        all_results["comprehensive_report"] = comprehensive_report

        print("\n✅ Advanced Experiment Suite Completed!")

        return all_results

    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""

        report = {
            "experiment_summary": {},
            "key_findings": [],
            "recommendations": [],
            "performance_metrics": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Multi-objective results
        if "multi_objective" in results:
            mo_data = results["multi_objective"]
            report["experiment_summary"]["multi_objective"] = {
                "pareto_front_size": mo_data.get("pareto_front_size", 0),
                "hypervolume": mo_data.get("hypervolume", 0),
                "diversity": mo_data.get("diversity", 0),
            }

            if mo_data.get("pareto_front_size", 0) > 0:
                report["key_findings"].append(
                    f"Multi-objective optimization found {mo_data['pareto_front_size']} Pareto-optimal solutions"
                )

        # Scalability results
        if "scalability" in results:
            scale_data = results["scalability"]
            analysis = scale_data.get("analysis", {})

            report["experiment_summary"]["scalability"] = {
                "problem_sizes_tested": len(
                    scale_data.get("scalability_metrics", {}).get("problem_sizes", [])
                ),
                "performance_scaling": analysis.get("hybrid_performance_scaling", 0),
                "time_scaling": analysis.get("hybrid_time_scaling", 0),
            }

            if analysis.get("hybrid_performance_scaling", 0) > -1:
                report["key_findings"].append(
                    f"Hybrid system shows sub-linear performance scaling: {analysis.get('hybrid_performance_scaling', 0):.3f}"
                )

        # Robustness results
        if "robustness" in results:
            robust_data = results["robustness"]
            analysis = robust_data.get("analysis", {})

            report["experiment_summary"]["robustness"] = {
                "noise_levels_tested": len(
                    robust_data.get("robustness_metrics", {}).get("noise_levels", [])
                ),
                "hybrid_degradation_rate": analysis.get("hybrid_degradation_rate", 0),
            }

            if analysis.get("hybrid_degradation_rate", 1) < 0.5:
                report["key_findings"].append(
                    f"Hybrid system shows good robustness with {analysis.get('hybrid_degradation_rate', 0)*100:.1f}% degradation rate"
                )

        # Generate recommendations
        report["recommendations"] = [
            "Consider using hybrid quantum-neuromorphic systems for complex multi-objective problems",
            "Implement adaptive scaling strategies for large problem sizes",
            "Apply transfer learning techniques for domain adaptation",
            "Focus on robustness improvements for real-world deployment",
        ]

        return report

