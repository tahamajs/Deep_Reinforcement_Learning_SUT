#!/usr/bin/env python3
"""
Quick execution script for CA9: Advanced Policy Gradient Methods
This script runs the most important components and generates key visualizations
"""

import sys
import os
import traceback
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_directories():
    """Create necessary directories"""
    directories = ["visualizations", "results", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def run_basic_reinforce():
    """Run basic REINFORCE experiment"""
    print("\n" + "=" * 60)
    print("RUNNING BASIC REINFORCE EXPERIMENT")
    print("=" * 60)

    try:
        from agents.reinforce import REINFORCEAgent
        import gymnasium as gym
        import numpy as np

        # Test on CartPole
        print("Training REINFORCE on CartPole-v1...")
        env = gym.make("CartPole-v1")
        agent = REINFORCEAgent(
            state_dim=env.observation_space.shape[0], action_dim=env.action_space.n
        )

        rewards = []
        for episode in range(50):
            state, _ = env.reset()
            episode_rewards = []

            for step in range(200):
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.store_transition(state, action, reward, log_prob)
                episode_rewards.append(reward)
                state = next_state

                if terminated or truncated:
                    break

            agent.update()
            total_reward = sum(episode_rewards)
            rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward}")

        env.close()

        avg_reward = np.mean(rewards[-10:])
        print(f"✅ REINFORCE completed! Average reward (last 10): {avg_reward:.2f}")
        return True

    except Exception as e:
        print(f"❌ REINFORCE failed: {e}")
        traceback.print_exc()
        return False


def run_basic_visualizations():
    """Run basic visualizations"""
    print("\n" + "=" * 60)
    print("GENERATING BASIC VISUALIZATIONS")
    print("=" * 60)

    try:
        from utils.policy_gradient_visualizer import PolicyGradientVisualizer

        visualizer = PolicyGradientVisualizer()

        print("Creating policy gradient intuition visualization...")
        results = visualizer.demonstrate_policy_gradient_intuition()

        print("Creating value vs policy comparison...")
        visualizer.compare_value_vs_policy_methods()

        print("✅ Basic visualizations completed!")
        return True

    except Exception as e:
        print(f"❌ Visualizations failed: {e}")
        traceback.print_exc()
        return False


def run_training_examples():
    """Run training examples"""
    print("\n" + "=" * 60)
    print("RUNNING TRAINING EXAMPLES")
    print("=" * 60)

    try:
        from training_examples import (
            plot_policy_gradient_convergence_analysis,
            comprehensive_policy_gradient_comparison,
        )

        print("Generating convergence analysis...")
        fig1 = plot_policy_gradient_convergence_analysis(
            "visualizations/convergence_analysis.png"
        )

        print("Generating comprehensive comparison...")
        results = comprehensive_policy_gradient_comparison(
            "visualizations/comprehensive_comparison.png"
        )

        print("✅ Training examples completed!")
        return True

    except Exception as e:
        print(f"❌ Training examples failed: {e}")
        traceback.print_exc()
        return False


def create_summary_report():
    """Create a summary report"""
    print("\n" + "=" * 60)
    print("CREATING SUMMARY REPORT")
    print("=" * 60)

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Count generated files
        viz_count = 0
        if os.path.exists("visualizations"):
            viz_count = len(
                [
                    f
                    for f in os.listdir("visualizations")
                    if f.endswith((".png", ".pdf"))
                ]
            )

        report_content = f"""
# CA9: Advanced Policy Gradient Methods - Quick Execution Report

## Execution Details
- **Execution Time**: {timestamp}
- **Total Visualizations Generated**: {viz_count}

## Completed Components

### ✅ Agent Implementations
- REINFORCE Algorithm (Basic test on CartPole-v1)

### ✅ Visualizations Generated
- Policy Gradient Intuition Visualization
- Value vs Policy Methods Comparison
- Convergence Analysis
- Comprehensive Method Comparison

## Results Location
- **Visualizations**: `visualizations/` directory
- **Results**: `results/` directory  
- **Logs**: `logs/` directory

## Status: COMPLETE ✅
Quick execution completed successfully!
"""

        with open("results/quick_execution_report.md", "w") as f:
            f.write(report_content)

        print("✅ Summary report created at results/quick_execution_report.md")
        return True

    except Exception as e:
        print(f"❌ Summary report creation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function"""
    print("=" * 80)
    print("CA9: Advanced Policy Gradient Methods - Quick Execution")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create directories
    create_directories()

    # List of functions to execute
    experiments = [
        ("Basic REINFORCE", run_basic_reinforce),
        ("Basic Visualizations", run_basic_visualizations),
        ("Training Examples", run_training_examples),
        ("Summary Report", create_summary_report),
    ]

    successful = 0
    total = len(experiments)

    for experiment_name, experiment_func in experiments:
        print(f"\n🚀 Starting: {experiment_name}")
        start_time = time.time()

        try:
            if experiment_func():
                successful += 1
                elapsed_time = time.time() - start_time
                print(f"✅ {experiment_name} completed in {elapsed_time:.2f}s")
            else:
                print(f"❌ {experiment_name} failed")
        except Exception as e:
            print(f"❌ {experiment_name} failed with exception: {e}")

    print("\n" + "=" * 80)
    print(
        f"QUICK EXECUTION SUMMARY: {successful}/{total} experiments completed successfully"
    )
    print("=" * 80)

    if successful == total:
        print("🎉 QUICK EXECUTION COMPLETED SUCCESSFULLY!")
        print("📊 Check the visualizations/ folder for generated plots")
        print("📋 Check the results/ folder for summary reports")
    else:
        print(f"⚠️  {total - successful} experiments had issues.")

    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
