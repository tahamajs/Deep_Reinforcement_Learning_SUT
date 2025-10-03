"""
CA5 Deep Q-Networks: Complete Implementation and Analysis
========================================================

This module provides a complete, modular implementation of Deep Q-Networks
with all major variants and comprehensive analysis tools.

Project Structure:
- dqn_base.py: Core DQN implementation with experience replay
- double_dqn.py: Double DQN with overestimation bias correction
- dueling_dqn.py: Dueling architecture separating value and advantage
- prioritized_replay.py: Prioritized Experience Replay for sample efficiency
- rainbow_dqn.py: Rainbow DQN combining all improvements
- analysis_tools.py: Comprehensive analysis and visualization tools

Key Features:
- Modular architecture for easy extension
- Comprehensive analysis tools
- Performance comparison framework
- Hyperparameter sensitivity analysis
- Learning dynamics visualization
- Statistical significance testing

Usage Examples:
- Basic DQN training
- Comparative analysis of variants
- Hyperparameter optimization
- Algorithm behavior analysis

Author: CA5 Implementation
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from dqn_base import DQNAgent
from double_dqn import DoubleDQNAgent, OverestimationAnalysis, DQNComparison
from dueling_dqn import DuelingDQNAgent, DuelingAnalysis
from prioritized_replay import PrioritizedDQNAgent, PERAnalysis
from rainbow_dqn import RainbowDQNAgent, RainbowAnalysis
from analysis_tools import (
    DQNComparator,
    HyperparameterAnalyzer,
    LearningDynamicsAnalyzer,
    PerformanceProfiler,
)
import warnings

warnings.filterwarnings("ignore")


class CA5Demonstrator:
    """Demonstration class for CA5 DQN implementations"""

    def __init__(self, env_name="CartPole-v1"):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        print(f"Environment: {env_name}")
        print(f"State size: {self.state_size}")
        print(f"Action size: {self.action_size}")
        print()

    def demonstrate_basic_dqn(self):
        """Demonstrate basic DQN training"""
        print("1. Basic DQN Demonstration")
        print("=" * 30)

        agent = DQNAgent(self.state_size, self.action_size)
        scores, training_info = agent.train(self.env, num_episodes=100, print_every=25)

        print(f"Final average score: {np.mean(scores[-20:]):.2f}")
        print(f"Training completed in {training_info['total_time']:.2f} seconds")
        print()

        return agent, scores

    def demonstrate_double_dqn(self):
        """Demonstrate Double DQN with bias analysis"""
        print("2. Double DQN Demonstration")
        print("=" * 32)

        agent = DoubleDQNAgent(self.state_size, self.action_size)
        scores, _ = agent.train(self.env, num_episodes=100, print_every=25)

        bias_analysis = OverestimationAnalysis()
        bias_results = bias_analysis.visualize_bias_analysis()

        print(f"Double DQN final average score: {np.mean(scores[-20:]):.2f}")
        print()

        return agent, scores, bias_results

    def demonstrate_dueling_dqn(self):
        """Demonstrate Dueling DQN architecture"""
        print("3. Dueling DQN Demonstration")
        print("=" * 32)

        agent = DuelingDQNAgent(self.state_size, self.action_size)
        scores, _ = agent.train(self.env, num_episodes=100, print_every=25)

        dueling_analysis = DuelingAnalysis()
        dueling_analysis.visualize_dueling_architecture(agent, self.env)

        print(f"Dueling DQN final average score: {np.mean(scores[-20:]):.2f}")
        print()

        return agent, scores

    def demonstrate_prioritized_replay(self):
        """Demonstrate Prioritized Experience Replay"""
        print("4. Prioritized Experience Replay Demonstration")
        print("=" * 48)

        agent = PrioritizedDQNAgent(self.state_size, self.action_size)
        scores, _ = agent.train(self.env, num_episodes=100, print_every=25)

        per_analysis = PERAnalysis()
        per_analysis.visualize_per_behavior(agent)

        print(f"PER DQN final average score: {np.mean(scores[-20:]):.2f}")
        print()

        return agent, scores

    def demonstrate_rainbow_dqn(self):
        """Demonstrate Rainbow DQN (all improvements combined)"""
        print("5. Rainbow DQN Demonstration")
        print("=" * 32)

        agent = RainbowDQNAgent(self.state_size, self.action_size)
        scores, _ = agent.train(self.env, num_episodes=100, print_every=25)

        rainbow_analysis = RainbowAnalysis()
        rainbow_analysis.visualize_rainbow_components(agent, self.env)

        print(f"Rainbow DQN final average score: {np.mean(scores[-20:]):.2f}")
        print()

        return agent, scores

    def run_comprehensive_comparison(self):
        """Run comprehensive comparison of all DQN variants"""
        print("6. Comprehensive DQN Comparison")
        print("=" * 38)

        comparator = DQNComparator(self.env, self.state_size, self.action_size)

        comparator.add_agent("DQN", DQNAgent, lr=0.001)
        comparator.add_agent("Double DQN", DoubleDQNAgent, lr=0.001)
        comparator.add_agent("Dueling DQN", DuelingDQNAgent, lr=0.001)
        comparator.add_agent("Prioritized DQN", PrioritizedDQNAgent, lr=0.001)
        comparator.add_agent("Rainbow DQN", RainbowDQNAgent, lr=0.001)

        results = comparator.run_comparison(num_episodes=200, num_runs=2)

        comparator.visualize_comparison()

        comparator.statistical_analysis()

        return results

    def demonstrate_hyperparameter_analysis(self):
        """Demonstrate hyperparameter sensitivity analysis"""
        print("7. Hyperparameter Sensitivity Analysis")
        print("=" * 42)

        analyzer = HyperparameterAnalyzer(
            self.env, self.state_size, self.action_size, DQNAgent
        )

        lr_results = analyzer.analyze_learning_rate(
            learning_rates=[0.0001, 0.001, 0.01], num_episodes=100, num_runs=1
        )

        batch_results = analyzer.analyze_batch_size(
            batch_sizes=[16, 64, 128], num_episodes=100, num_runs=1
        )

        print("Hyperparameter analysis completed.")
        print()

        return lr_results, batch_results

    def demonstrate_performance_profiling(self):
        """Demonstrate computational performance profiling"""
        print("8. Performance Profiling")
        print("=" * 26)

        profiler = PerformanceProfiler()

        agent_configs = {
            DQNAgent: {"lr": 0.001},
            DoubleDQNAgent: {"lr": 0.001},
            DuelingDQNAgent: {"lr": 0.001},
            PrioritizedDQNAgent: {"lr": 0.001},
            RainbowDQNAgent: {"lr": 0.001},
        }

        profiler.compare_performance_profiles(agent_configs)

        print("Performance profiling completed.")
        print()

    def run_full_demonstration(self):
        """Run complete demonstration of all CA5 components"""
        print("CA5 Deep Q-Networks: Complete Demonstration")
        print("=" * 50)
        print()

        basic_agent, basic_scores = self.demonstrate_basic_dqn()
        double_agent, double_scores, bias_results = self.demonstrate_double_dqn()
        dueling_agent, dueling_scores = self.demonstrate_dueling_dqn()
        per_agent, per_scores = self.demonstrate_prioritized_replay()
        rainbow_agent, rainbow_scores = self.demonstrate_rainbow_dqn()

        print("9. Summary Comparison")
        print("=" * 23)

        agents_scores = [
            ("Basic DQN", basic_scores),
            ("Double DQN", double_scores),
            ("Dueling DQN", dueling_scores),
            ("Prioritized DQN", per_scores),
            ("Rainbow DQN", rainbow_scores),
        ]

        print("Final Performance Summary:")
        print("-" * 30)
        for name, scores in agents_scores:
            final_avg = np.mean(scores[-20:])
            improvement = final_avg - np.mean(scores[:20])
            print("<15")

        print()
        print("✓ CA5 Complete demonstration finished")
        print("✓ All DQN variants implemented and tested")
        print("✓ Comprehensive analysis completed")


def create_usage_examples():
    """Create usage examples for different scenarios"""

    examples = {
        "basic_training": """

from dqn_base import DQNAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = DQNAgent(state_size=4, action_size=2)
scores, info = agent.train(env, num_episodes=500)
print(f"Training completed. Final score: {np.mean(scores[-50:]):.2f}")
        """,
        "comparison_study": """

from analysis_tools import DQNComparator
from dqn_base import DQNAgent
from double_dqn import DoubleDQNAgent

env = gym.make("CartPole-v1")
comparator = DQNComparator(env, 4, 2)

comparator.add_agent("DQN", DQNAgent)
comparator.add_agent("Double DQN", DoubleDQNAgent)

results = comparator.run_comparison(num_episodes=300, num_runs=3)
comparator.visualize_comparison()
comparator.statistical_analysis()
        """,
        "hyperparameter_tuning": """

from analysis_tools import HyperparameterAnalyzer
from dueling_dqn import DuelingDQNAgent

env = gym.make("CartPole-v1")
analyzer = HyperparameterAnalyzer(env, 4, 2, DuelingDQNAgent)


lr_results = analyzer.analyze_learning_rate([0.0001, 0.001, 0.01])
        """,
        "advanced_analysis": """

from double_dqn import OverestimationAnalysis
from dueling_dqn import DuelingAnalysis


bias_analyzer = OverestimationAnalysis()
bias_results = bias_analyzer.visualize_bias_analysis()


dueling_analyzer = DuelingAnalysis()


        """,
    }

    return examples


if __name__ == "__main__":
    print("CA5 Deep Q-Networks Implementation")
    print("=" * 40)
    print()

    demo = CA5Demonstrator()

    examples = create_usage_examples()
    print("Usage Examples:")
    print("=" * 16)
    for name, code in examples.items():
        print(f"\\n{name.upper()}:")
        print("-" * (len(name) + 1))
        print(code.strip())

    print("\\n" + "=" * 50)
    print("✓ CA5 Implementation Complete")
    print("✓ All modules ready for use")
    print("✓ Comprehensive DQN framework available")
    print("=" * 50)
