#!/usr/bin/env python3
"""
CA5 Advanced DQN Methods - Main Entry Point
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PrioritizedDQNAgent
from environments import make_env
from training_examples import train_dqn_agent, dqn_variant_comparison
from experiments import ExperimentRunner, get_dqn_configs
from evaluation import PerformanceEvaluator, compare_agents


def main():
    """Main entry point for CA5 Advanced DQN Methods"""

    parser = argparse.ArgumentParser(
        description="CA5 Advanced DQN Methods - Complete Implementation"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "compare", "experiment", "evaluate", "all"],
        default="all",
        help="Execution mode",
    )

    parser.add_argument("--env", default="CartPole-v1", help="Environment name")

    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )

    parser.add_argument(
        "--agent",
        choices=["dqn", "double", "dueling", "prioritized", "rainbow"],
        default="dqn",
        help="Agent type",
    )

    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    print("=" * 60)
    print("CA5 Advanced DQN Methods - Main Execution")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Agent: {args.agent}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    results = {}

    if args.mode in ["train", "all"]:
        print("\n🚀 Starting Training...")
        try:
            training_results = train_dqn_agent(
                env_name=args.env, agent_type=args.agent, num_episodes=args.episodes
            )
            results["training"] = training_results

            # Save training results
            with open(f"{args.output_dir}/training_results.json", "w") as f:
                json.dump(training_results, f, indent=2)

            print(
                f"✅ Training completed! Final average reward: {training_results['final_avg_reward']:.2f}"
            )

        except Exception as e:
            print(f"❌ Training failed: {e}")

    if args.mode in ["compare", "all"]:
        print("\n� Starting Agent Comparison...")
        try:
            comparison_results = dqn_variant_comparison()
            results["comparison"] = comparison_results

            # Save comparison results
            with open(f"{args.output_dir}/comparison_results.json", "w") as f:
                json.dump(comparison_results, f, indent=2)

            print("✅ Agent comparison completed!")

        except Exception as e:
            print(f"❌ Comparison failed: {e}")

    if args.mode in ["experiment", "all"]:
        print("\n🧪 Starting Experiments...")
        try:
            runner = ExperimentRunner()
            configs = get_dqn_configs()

            # Reduce episodes for demo
            for config in configs:
                config.config["training"]["num_episodes"] = min(args.episodes, 500)

            agent_classes = [
                DQNAgent,
                DoubleDQNAgent,
                DuelingDQNAgent,
                PrioritizedDQNAgent,
            ]

            experiment_results = runner.run_comparison_experiment(
                configs[:2], agent_classes[:2], args.env  # First 2 for demo
            )
            results["experiments"] = experiment_results

            print("✅ Experiments completed!")

        except Exception as e:
            print(f"❌ Experiments failed: {e}")

    if args.mode in ["evaluate", "all"]:
        print("\n📊 Starting Evaluation...")
        try:
            import gym

            env = gym.make(args.env)

            # Create agents for evaluation
            agents = {
                "DQN": DQNAgent(
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.n,
                    lr=1e-3,
                ),
                "Double DQN": DoubleDQNAgent(
                    state_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.n,
                    lr=1e-3,
                ),
            }

            # Quick training for evaluation
            for name, agent in agents.items():
                print(f"Quick training {name}...")
                for episode in range(min(args.episodes // 10, 100)):
                    state = env.reset()
                    done = False
                    while not done:
                        action = agent.select_action(state)
                        next_state, reward, done, info = env.step(action)
                        agent.replay_buffer.push(
                            state, action, reward, next_state, done
                        )
                        if len(agent.replay_buffer) > agent.batch_size:
                            agent.update()
                        state = next_state

            # Evaluate agents
            eval_results = compare_agents(agents, env, num_episodes=20)
            results["evaluation"] = eval_results

            env.close()
            print("✅ Evaluation completed!")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

    # Generate summary report
    print("\n📋 Generating Summary Report...")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "execution_mode": args.mode,
        "environment": args.env,
        "episodes": args.episodes,
        "agent_type": args.agent,
        "results_summary": {
            "training_completed": "training" in results,
            "comparison_completed": "comparison" in results,
            "experiments_completed": "experiments" in results,
            "evaluation_completed": "evaluation" in results,
        },
        "output_files": [
            f"{args.output_dir}/training_results.json",
            f"{args.output_dir}/comparison_results.json",
            f"{args.output_dir}/summary_report.json",
        ],
    }

    # Save summary
    with open(f"{args.output_dir}/summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("🎉 Execution Completed Successfully!")
    print("=" * 60)
    print(f"Results saved in: {args.output_dir}/")
    print(f"Visualizations saved in: visualizations/")
    print(f"Summary report: {args.output_dir}/summary_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
