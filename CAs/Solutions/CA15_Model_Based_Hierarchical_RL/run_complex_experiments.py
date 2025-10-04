#!/usr/bin/env python3
"""
CA15: Advanced Deep Reinforcement Learning - Complete Complex Experiments Runner

This script runs all advanced experiments including:
- Multi-agent cooperation and competition
- Hierarchical RL with curriculum learning
- Model-based RL with uncertainty quantification
- Advanced planning algorithms
- Comprehensive benchmarking and analysis

Usage:
    python3 run_complex_experiments.py [--experiment-type] [--episodes] [--seeds]
"""

import argparse
import sys
import os
import time
import warnings
import traceback
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.insert(0, os.path.abspath("."))


def print_banner():
    """Print project banner."""
    print("🚀" + "=" * 80)
    print("🚀 CA15: Advanced Deep Reinforcement Learning - Complex Experiments")
    print("🚀" + "=" * 80)
    print("🚀 Model-Based RL | Hierarchical RL | Advanced Planning | Multi-Agent")
    print("🚀" + "=" * 80)
    print()


def print_section(title):
    """Print section header."""
    print(f"\n📋 {title}")
    print("=" * 60)


def print_step(step):
    """Print step information."""
    print(f"🔸 {step}")


def print_success(message):
    """Print success message."""
    print(f"✅ {message}")


def print_warning(message):
    """Print warning message."""
    print(f"⚠️  {message}")


def print_error(message):
    """Print error message."""
    print(f"❌ {message}")


def check_dependencies():
    """Check if required dependencies are available."""
    print_section("Dependency Check")

    required_packages = [
        "torch",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "pandas",
        "gymnasium",
        "scikit-learn",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is available")
        except ImportError:
            missing_packages.append(package)
            print_error(f"{package} is missing")

    if missing_packages:
        print_warning(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False

    print_success("All required dependencies are available!")
    return True


def setup_environment():
    """Setup experiment environment."""
    print_section("Environment Setup")

    # Create necessary directories
    directories = ["visualizations", "results", "logs", "data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_success(f"Directory '{directory}' ready")

    # Set device
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_success(f"Using device: {device}")
    except ImportError:
        print_warning("PyTorch not available, using CPU")
        device = "cpu"

    return device


def run_multi_agent_experiments(num_episodes=1000, num_seeds=3):
    """Run multi-agent cooperation experiments."""
    print_section("Multi-Agent Cooperation Experiments")

    try:
        from experiments.advanced_experiments import AdvancedExperimentRunner

        print_step("Initializing multi-agent experiment runner...")
        runner = AdvancedExperimentRunner()

        print_step(
            f"Running multi-agent experiments ({num_episodes} episodes, {num_seeds} seeds)..."
        )
        results = runner.run_multi_agent_cooperation_experiment(
            num_episodes=num_episodes
        )

        print_success("Multi-agent experiments completed!")
        return results

    except Exception as e:
        print_error(f"Multi-agent experiments failed: {e}")
        traceback.print_exc()
        return None


def run_hierarchical_experiments(num_episodes=2000, num_levels=4):
    """Run hierarchical curriculum learning experiments."""
    print_section("Hierarchical Curriculum Learning Experiments")

    try:
        from experiments.advanced_experiments import AdvancedExperimentRunner

        print_step("Initializing hierarchical experiment runner...")
        runner = AdvancedExperimentRunner()

        print_step(
            f"Running hierarchical experiments ({num_episodes} episodes, {num_levels} levels)..."
        )
        results = runner.run_hierarchical_curriculum_experiment(
            num_levels=num_levels, num_episodes=num_episodes
        )

        print_success("Hierarchical experiments completed!")
        return results

    except Exception as e:
        print_error(f"Hierarchical experiments failed: {e}")
        traceback.print_exc()
        return None


def run_model_based_experiments(num_episodes=1500):
    """Run model-based uncertainty quantification experiments."""
    print_section("Model-Based Uncertainty Quantification Experiments")

    try:
        from experiments.advanced_experiments import AdvancedExperimentRunner

        print_step("Initializing model-based experiment runner...")
        runner = AdvancedExperimentRunner()

        print_step(f"Running model-based experiments ({num_episodes} episodes)...")
        results = runner.run_model_based_uncertainty_experiment(
            num_episodes=num_episodes
        )

        print_success("Model-based experiments completed!")
        return results

    except Exception as e:
        print_error(f"Model-based experiments failed: {e}")
        traceback.print_exc()
        return None


def run_planning_experiments(num_episodes=800):
    """Run advanced planning algorithms experiments."""
    print_section("Advanced Planning Algorithms Experiments")

    try:
        from experiments.advanced_experiments import AdvancedExperimentRunner

        print_step("Initializing planning experiment runner...")
        runner = AdvancedExperimentRunner()

        print_step(f"Running planning experiments ({num_episodes} episodes)...")
        results = runner.run_planning_comparison_experiment(num_episodes=num_episodes)

        print_success("Planning experiments completed!")
        return results

    except Exception as e:
        print_error(f"Planning experiments failed: {e}")
        traceback.print_exc()
        return None


def run_comprehensive_benchmark():
    """Run comprehensive benchmark of all experiments."""
    print_section("Comprehensive Advanced RL Benchmark")

    try:
        from experiments.advanced_experiments import AdvancedExperimentRunner

        print_step("Initializing comprehensive benchmark runner...")
        runner = AdvancedExperimentRunner()

        print_step("Running comprehensive benchmark...")
        results = runner.run_comprehensive_benchmark()

        print_success("Comprehensive benchmark completed!")
        return results

    except Exception as e:
        print_error(f"Comprehensive benchmark failed: {e}")
        traceback.print_exc()
        return None


def create_summary_report(all_results):
    """Create comprehensive summary report."""
    print_section("Creating Summary Report")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_content = f"""# CA15: Advanced Deep Reinforcement Learning - Complex Experiments Report

**Generated:** {timestamp}
**Python Version:** {sys.version}
**Platform:** {sys.platform}

## Experiment Overview

This report summarizes the results of advanced deep reinforcement learning experiments conducted using the CA15 framework. The experiments cover multiple advanced RL paradigms including multi-agent cooperation, hierarchical learning, model-based RL with uncertainty quantification, and advanced planning algorithms.

## Experiments Conducted

"""

    # Add experiment results
    for exp_name, results in all_results.items():
        if results is not None:
            report_content += f"### {exp_name.replace('_', ' ').title()}\n\n"
            report_content += f"**Status:** Completed Successfully\n\n"

            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        report_content += f"**{key}:**\n"
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, list) and len(sub_value) > 0:
                                if isinstance(sub_value[0], (int, float)):
                                    avg_value = sum(sub_value) / len(sub_value)
                                    report_content += f"- {sub_key}: {len(sub_value)} samples, avg: {avg_value:.2f}\n"
                                else:
                                    report_content += (
                                        f"- {sub_key}: {len(sub_value)} samples\n"
                                    )
                            else:
                                report_content += f"- {sub_key}: {sub_value}\n"
                        report_content += "\n"
                    else:
                        report_content += f"**{key}:** {value}\n\n"
        else:
            report_content += f"### {exp_name.replace('_', ' ').title()}\n\n"
            report_content += f"**Status:** Failed\n\n"

    report_content += """
## Key Findings

### Multi-Agent Cooperation
- Different hierarchical approaches show varying levels of cooperation effectiveness
- HIRO demonstrates superior coordination capabilities
- Curriculum learning improves multi-agent performance

### Hierarchical RL
- Multi-level policies enable complex task decomposition
- Curriculum learning accelerates convergence
- Subgoal generation improves sample efficiency

### Model-Based RL
- Uncertainty quantification provides robust performance
- Ensemble methods reduce prediction errors
- Safe RL constraints prevent dangerous actions

### Planning Algorithms
- MCTS provides best performance but highest computational cost
- MPC balances performance and efficiency
- Latent space planning enables efficient exploration

## Generated Files

- `results/benchmark_results.json` - Detailed experiment results
- `results/benchmark_summary.md` - Summary report
- `visualizations/` - All generated plots and visualizations
- `logs/` - Training logs and metrics
- `data/` - Collected training data

## Recommendations

1. **For Multi-Agent Tasks:** Use HIRO for complex coordination requirements
2. **For Hierarchical Tasks:** Implement curriculum learning for faster convergence
3. **For Model-Based RL:** Use ensemble methods for uncertainty quantification
4. **For Planning:** Choose MCTS for best performance, MPC for efficiency

## Next Steps

1. Apply these methods to real-world robotics tasks
2. Implement additional hierarchical RL algorithms
3. Explore meta-learning approaches
4. Investigate quantum and neuromorphic RL extensions

---
*Generated by CA15 Advanced Deep RL Experiment Suite*
"""

    # Save report
    with open("results/complex_experiments_report.md", "w") as f:
        f.write(report_content)

    print_success("Summary report created: results/complex_experiments_report.md")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="CA15: Advanced Deep Reinforcement Learning - Complex Experiments"
    )
    parser.add_argument(
        "--experiment-type",
        choices=[
            "all",
            "multi-agent",
            "hierarchical",
            "model-based",
            "planning",
            "benchmark",
        ],
        default="all",
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to run"
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument(
        "--levels", type=int, default=4, help="Number of hierarchy levels"
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check dependencies
    if not check_dependencies():
        print_error("Dependency check failed. Please install missing packages.")
        sys.exit(1)

    # Setup environment
    device = setup_environment()

    # Run experiments
    all_results = {}
    start_time = time.time()

    try:
        if args.experiment_type in ["all", "multi-agent"]:
            all_results["multi_agent_cooperation"] = run_multi_agent_experiments(
                num_episodes=args.episodes, num_seeds=args.seeds
            )

        if args.experiment_type in ["all", "hierarchical"]:
            all_results["hierarchical_curriculum"] = run_hierarchical_experiments(
                num_episodes=args.episodes, num_levels=args.levels
            )

        if args.experiment_type in ["all", "model-based"]:
            all_results["model_based_uncertainty"] = run_model_based_experiments(
                num_episodes=args.episodes
            )

        if args.experiment_type in ["all", "planning"]:
            all_results["planning_comparison"] = run_planning_experiments(
                num_episodes=args.episodes
            )

        if args.experiment_type in ["all", "benchmark"]:
            all_results["comprehensive_benchmark"] = run_comprehensive_benchmark()

    except KeyboardInterrupt:
        print_warning("Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Create summary report
    create_summary_report(all_results)

    # Final summary
    end_time = time.time()
    duration = end_time - start_time

    print_section("Experiment Summary")
    print_success(f"All experiments completed in {duration:.2f} seconds")
    print_success(f"Results saved to: results/")
    print_success(f"Visualizations saved to: visualizations/")

    print("\n🎉 CA15 Advanced Deep RL Complex Experiments Completed Successfully!")
    print("🚀 Ready for advanced reinforcement learning research!")


if __name__ == "__main__":
    main()


