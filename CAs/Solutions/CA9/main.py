#!/usr/bin/env python3
"""
Policy Gradient Methods Demonstration
=====================================

This script demonstrates various policy gradient methods for reinforcement learning,
including REINFORCE, Actor-Critic, and PPO algorithms.

Usage:
    python main.py

The script will run demonstrations of different policy gradient algorithms
and display their performance comparisons.
"""

from utils import test_environment_setup
from policy_gradient_visualizer import PolicyGradientVisualizer
from reinforce import REINFORCEAnalyzer
from baseline_reinforce import VarianceAnalyzer
from actor_critic import ActorCriticAnalyzer
from ppo import AdvancedPolicyGradientAnalyzer


def main():
    """Main demonstration function"""

    print("🎓 Policy Gradient Methods in Reinforcement Learning")
    print("=" * 60)

    # Test environment setup
    print("\n1. Testing Environment Setup...")
    test_environment_setup()

    # Policy gradient intuition
    print("\n2. Policy Gradient Intuition...")
    pg_visualizer = PolicyGradientVisualizer()
    intuition_results = pg_visualizer.demonstrate_policy_gradient_intuition()

    # Value-based vs Policy-based comparison
    print("\n3. Value-based vs Policy-based Methods Comparison...")
    pg_visualizer.compare_value_vs_policy_methods()

    # REINFORCE algorithm
    print("\n4. REINFORCE Algorithm Training...")
    reinforce_analyzer = REINFORCEAnalyzer()
    reinforce_agent = reinforce_analyzer.train_and_analyze(
        "CartPole-v1", num_episodes=300
    )

    # Variance reduction techniques
    print("\n5. Variance Reduction Techniques...")
    variance_analyzer = VarianceAnalyzer()
    variance_results = variance_analyzer.compare_baseline_methods(
        "CartPole-v1", num_episodes=250
    )

    # Actor-Critic methods
    print("\n6. Actor-Critic Methods Comparison...")
    ac_analyzer = ActorCriticAnalyzer()
    ac_results = ac_analyzer.compare_actor_critic_variants(
        "CartPole-v1", num_episodes=250
    )

    # Comprehensive comparison
    print("\n7. Comprehensive Policy Gradient Methods Comparison...")
    advanced_analyzer = AdvancedPolicyGradientAnalyzer()
    comprehensive_results = advanced_analyzer.compare_all_methods(
        "CartPole-v1", num_episodes=200
    )

    print("\n" + "=" * 60)
    print("🎉 All demonstrations completed successfully!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("• Policy gradient methods directly optimize the policy")
    print("• Variance reduction techniques improve stability")
    print("• Actor-Critic methods combine policy and value learning")
    print("• PPO provides state-of-the-art performance with stability")
    print("• More advanced methods generally offer better sample efficiency")


if __name__ == "__main__":
    main()
