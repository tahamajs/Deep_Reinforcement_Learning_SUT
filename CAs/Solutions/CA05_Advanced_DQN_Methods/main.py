#!/usr/bin/env python3
"""
CA5 Advanced DQN Methods - Main Entry Point
"""

import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports from refactored modules
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.dqn_base import DQNAgent
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.double_dqn import DoubleDQNAgent
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.dueling_dqn import DuelingDQNAgent
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.prioritized_replay_dqn import PrioritizedDQNAgent
from CAs.Solutions.CA05_Advanced_DQN_Methods.environments import make_env # Assuming make_env is still relevant
from CAs.Solutions.CA05_Advanced_DQN_Methods.training_examples import train_dqn_agent, dqn_variant_comparison
from CAs.Solutions.CA05_Advanced_DQN_Methods.experiments.config import AgentConfig, ExperimentConfig, get_dqn_configs
# from evaluation import PerformanceEvaluator, compare_agents # Temporarily comment out if not yet refactored


def main():
    """Main entry point for CA5 Advanced DQN Methods"""

    parser = argparse.ArgumentParser(
        description="CA5 Advanced DQN Methods - Complete Implementation"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "compare", "all"],
        default="all",
        help="Execution mode",
    )

    parser.add_argument("--env", default="CartPole-v1", help="Environment name")

    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )

    parser.add_argument(
        "--agent",
        choices=["dqn", "double_dqn", "dueling_dqn", "prioritized_dqn", "rainbow_dqn"],
        default="dqn",
        help="Agent type for single training mode",
    )

    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--plots-dir", default="visualizations", help="Output directory for plots"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create output directories
    results_path = os.path.join(os.path.dirname(__file__), args.output_dir)
    plots_path = os.path.join(os.path.dirname(__file__), args.plots_dir)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    print("=" * 60)
    print("CA5 Advanced DQN Methods - Main Execution")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Agent for training: {args.agent}")
    print(f"Results Directory: {results_path}")
    print(f"Plots Directory: {plots_path}")
    print(f"Random Seed: {args.seed}")
    print("=" * 60)

    results = {}

    # Common experiment configuration
    base_experiment_config = ExperimentConfig(
        env_name=args.env,
        num_episodes=args.episodes,
        seed=args.seed,
        results_path=results_path,
        plots_path=plots_path,
    )

    if args.mode in ["train", "all"]:
        print("\n🚀 Starting Single Agent Training...")
        try:
            # Get agent specific config
            dqn_agent_configs = get_dqn_configs(args.env)
            if args.agent not in dqn_agent_configs:
                raise ValueError(f"Agent type {args.agent} not configured for {args.env}")

            agent_config = dqn_agent_configs[args.agent]
            # Override any CLI args if needed in agent_config or set them from CLI
            # For now, agent_config values are used primarily.

            training_results = train_dqn_agent(
                env_name=base_experiment_config.env_name,
                agent_type=args.agent,
                num_episodes=base_experiment_config.num_episodes,
                agent_config=agent_config,
                seed=base_experiment_config.seed,
            )
            results["training"] = training_results

            # Save training results
            training_output_file = os.path.join(results_path, f"{args.agent}_{args.env}_training_results.json")
            with open(training_output_file, "w") as f:
                json.dump({k: np.array(v).tolist() for k, v in training_results.items()}, f, indent=2)

            final_avg_reward = np.mean(training_results['episode_rewards'][-100:]) if training_results['episode_rewards'] else 0.0
            print(f"✅ Training completed! Final average reward: {final_avg_reward:.2f}"
            )
            print(f"Training results saved to: {training_output_file}")

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

    if args.mode in ["compare", "all"]:
        print("\n🔬 Starting Agent Comparison...")
        try:
            comparison_results = dqn_variant_comparison(
                env_name=base_experiment_config.env_name,
                num_episodes=base_experiment_config.num_episodes,
                num_runs=3, # Default to 3 runs for comparison
                save_path_prefix=os.path.join(base_experiment_config.plots_path, "comparison"),
            )
            results["comparison"] = comparison_results

            # Save comparison results
            comparison_output_file = os.path.join(results_path, f"{args.env}_comparison_results.json")
            # Convert numpy arrays in results to list for JSON serialization
            serializable_comparison_results = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                for k, v in comparison_results["avg_rewards_per_variant"].items()
            }
            # Simplified for now, full conversion would be more complex
            with open(comparison_output_file, "w") as f:
                json.dump(serializable_comparison_results, f, indent=2)

            print("✅ Agent comparison completed!")
            print(f"Comparison results saved to: {comparison_output_file}")

        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    print("\n📋 Generating Summary Report...")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "execution_mode": args.mode,
        "environment": args.env,
        "episodes": args.episodes,
        "agent_type": args.agent,
        "seed": args.seed,
        "results_summary": {
            "training_completed": "training" in results,
            "comparison_completed": "comparison" in results,
        },
        "output_files": [
            os.path.join(results_path, f"{args.agent}_{args.env}_training_results.json") if "training" in results else "N/A",
            os.path.join(results_path, f"{args.env}_comparison_results.json") if "comparison" in results else "N/A",
            os.path.join(results_path, "summary_report.json"),
        ],
        "plot_files": [
            os.path.join(plots_path, "comparison_comparison.png") if "comparison" in results else "N/A",
            # Additional plots would be listed here if generated in `training_examples.py` and saved
        ]
    }

    # Save summary
    summary_output_file = os.path.join(results_path, "summary_report.json")
    with open(summary_output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("🎉 Execution Completed Successfully!")
    print("=" * 60)
    print(f"Results saved in: {results_path}/")
    print(f"Plots saved in: {plots_path}/")
    print(f"Summary report: {summary_output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()


