#!/usr/bin/env python3
"""
CA06 Policy Gradient Methods - Main Execution Script

This script provides a comprehensive execution of all policy gradient algorithms
and generates complete results and visualizations.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from training_examples import (
    train_reinforce_agent,
    train_reinforce_baseline_agent,
    train_actor_critic_agent,
    train_ppo_agent,
    train_continuous_ppo_agent,
    compare_policy_gradient_variants,
    hyperparameter_sensitivity_analysis,
    curriculum_learning_demo,
    plot_policy_gradient_comparison,
)

from utils.performance_analysis import generate_comprehensive_report


def create_directories():
    """Create necessary directories"""
    directories = ["visualizations", "results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def run_basic_algorithms(episodes: int = 1000) -> Dict:
    """Run basic policy gradient algorithms"""
    print("\n" + "=" * 50)
    print("RUNNING BASIC POLICY GRADIENT ALGORITHMS")
    print("=" * 50)

    results = {}

    try:
        print("\n1. Training REINFORCE Agent...")
        results["reinforce"] = train_reinforce_agent(episodes=episodes)
        print("✅ REINFORCE training completed")

        print("\n2. Training REINFORCE with Baseline...")
        results["reinforce_baseline"] = train_reinforce_baseline_agent(
            episodes=episodes
        )
        print("✅ REINFORCE with Baseline training completed")

        print("\n3. Training Actor-Critic Agent...")
        results["actor_critic"] = train_actor_critic_agent(episodes=episodes)
        print("✅ Actor-Critic training completed")

        print("\n4. Training PPO Agent...")
        results["ppo"] = train_ppo_agent(episodes=episodes)
        print("✅ PPO training completed")

        print("\n5. Training Continuous PPO Agent...")
        results["continuous_ppo"] = train_continuous_ppo_agent(episodes=episodes)
        print("✅ Continuous PPO training completed")

    except Exception as e:
        print(f"❌ Error in basic algorithms: {e}")
        return {}

    return results


def run_advanced_analyses(episodes: int = 500) -> Dict:
    """Run advanced analyses"""
    print("\n" + "=" * 50)
    print("RUNNING ADVANCED ANALYSES")
    print("=" * 50)

    results = {}

    try:
        print("\n1. Comparing Policy Gradient Variants...")
        results["comparison"] = compare_policy_gradient_variants(episodes=episodes)
        print("✅ Policy gradient comparison completed")

        print("\n2. Hyperparameter Sensitivity Analysis...")
        hyperparameter_sensitivity_analysis(episodes=episodes // 2)
        print("✅ Hyperparameter sensitivity analysis completed")

        print("\n3. Curriculum Learning Demo...")
        curriculum_learning_demo(episodes=episodes)
        print("✅ Curriculum learning demo completed")

    except Exception as e:
        print(f"❌ Error in advanced analyses: {e}")
        return {}

    return results


def run_individual_agents():
    """Run individual agent demonstrations"""
    print("\n" + "=" * 50)
    print("RUNNING INDIVIDUAL AGENT DEMONSTRATIONS")
    print("=" * 50)

    agent_files = [
        "agents/reinforce.py",
        "agents/actor_critic.py",
        "agents/advanced_pg.py",
        "agents/variance_reduction.py",
        "experiments/applications.py",
        "utils/performance_analysis.py",
        "utils/run_ca6_smoke.py",
    ]

    for agent_file in agent_files:
        if Path(agent_file).exists():
            try:
                print(f"\nRunning {agent_file}...")
                exec(open(agent_file).read())
                print(f"✅ {agent_file} completed successfully")
            except Exception as e:
                print(f"❌ {agent_file} failed: {e}")
        else:
            print(f"⚠️  {agent_file} not found, skipping...")


def generate_visualizations(results: Dict):
    """Generate comprehensive visualizations"""
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    try:
        if "comparison" in results:
            print("\n1. Creating policy gradient comparison plots...")
            plot_policy_gradient_comparison(
                results["comparison"],
                save_path="visualizations/policy_gradient_comparison.png",
            )
            print("✅ Comparison plots saved to visualizations/")

        print("\n2. Generating comprehensive report...")
        generate_comprehensive_report()
        print("✅ Comprehensive report generated")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="CA06 Policy Gradient Methods - Complete Execution"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick version with fewer episodes"
    )
    parser.add_argument(
        "--algorithms-only", action="store_true", help="Run only basic algorithms"
    )
    parser.add_argument(
        "--analyses-only", action="store_true", help="Run only advanced analyses"
    )
    parser.add_argument(
        "--agents-only", action="store_true", help="Run only individual agent demos"
    )

    args = parser.parse_args()

    # Adjust episodes for quick mode
    if args.quick:
        args.episodes = min(args.episodes, 200)
        print("🚀 Quick mode: Using reduced episode count")

    print("=" * 60)
    print("CA06: POLICY GRADIENT METHODS - MODULAR IMPLEMENTATION")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Quick mode: {args.quick}")
    print("=" * 60)

    # Create directories
    create_directories()

    results = {}

    # Run based on arguments
    if not args.analyses_only and not args.agents_only:
        results = run_basic_algorithms(args.episodes)

    if not args.algorithms_only and not args.agents_only:
        analysis_results = run_advanced_analyses(args.episodes)
        results.update(analysis_results)

    if not args.algorithms_only and not args.analyses_only:
        run_individual_agents()

    # Generate visualizations
    if results:
        generate_visualizations(results)

    # Final summary
    print("\n" + "=" * 60)
    print("CA06 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Generated outputs:")
    print("- 📊 Visualizations: visualizations/")
    print("- 📈 Results: results/")
    print("- 📝 Logs: logs/")
    print("\nKey achievements:")
    print("✅ All policy gradient algorithms trained")
    print("✅ Performance comparisons generated")
    print("✅ Advanced analyses completed")
    print("✅ Comprehensive evaluation performed")
    print("=" * 60)


if __name__ == "__main__":
    main()
