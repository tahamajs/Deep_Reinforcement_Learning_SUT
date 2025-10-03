"""
Planning Algorithms Experiments

This module contains experiments for comparing different planning approaches
including Monte Carlo Tree Search, model-based value expansion, and latent space planning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlanningAlgorithmsExperiment:
    """Compare different planning approaches."""

    def __init__(self):
        self.results = {}

    def run_planning_comparison(self, num_episodes=200, num_seeds=2):
        """Compare MCTS, MVE, and random planning."""

        print("🎯 Running Planning Algorithms Comparison...")
        print("⚡ Testing: MCTS vs Model-Based Value Expansion vs Random Shooting")

        from ..environments.grid_world import SimpleGridWorld
        from ..model_based_rl.algorithms import DynaQAgent, ModelEnsemble
        from ..planning.algorithms import MonteCarloTreeSearch, ModelBasedValueExpansion

        env_size = 6

        state_dim = env_size * env_size
        action_dim = 4

        results = {}

        planning_configs = {
            "Random Shooting": {
                "use_mcts": False,
                "use_mve": False,
                "use_random": True,
            },
            "Model-Based Value Expansion": {
                "use_mcts": False,
                "use_mve": True,
                "use_random": False,
            },
            "MCTS Planning": {"use_mcts": True, "use_mve": False, "use_random": False},
        }

        for planner_name, config in planning_configs.items():
            print(f"\n🔄 Testing {planner_name}...")
            planner_results = []

            for seed in range(num_seeds):
                print(f"  Seed {seed + 1}/{num_seeds}")

                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)

                env = SimpleGridWorld(size=env_size)
                base_agent = DynaQAgent(state_dim, action_dim)

                model_ensemble = ModelEnsemble(state_dim, action_dim, ensemble_size=3)

                if config["use_mcts"]:
                    value_net = nn.Sequential(
                        nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)
                    ).to(device)
                    mcts_planner = MonteCarloTreeSearch(model_ensemble, value_net)
                    planner = mcts_planner
                elif config["use_mve"]:
                    value_net = base_agent.q_network
                    mve_planner = ModelBasedValueExpansion(model_ensemble, value_net)
                    planner = mve_planner
                else:
                    from ..model_based_rl.algorithms import ModelPredictiveController

                    mpc_planner = ModelPredictiveController(model_ensemble, action_dim)
                    planner = mpc_planner

                episode_rewards = []
                planning_times = []
                model_accuracy = []

                for episode in range(num_episodes):
                    state = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    done = False

                    while not done and episode_length < 100:
                        start_time = time.time()

                        if episode > 50:  # Start planning after some model training
                            try:
                                if config["use_mcts"]:
                                    root = planner.search(state, num_simulations=20)
                                    action_probs = planner.get_action_probabilities(
                                        root
                                    )
                                    action = np.argmax(action_probs)
                                elif config["use_mve"]:
                                    action = planner.plan_action(state)
                                else:
                                    action = planner.plan_action(state)
                            except:
                                action = base_agent.get_action(state, epsilon=0.1)
                        else:
                            action = base_agent.get_action(state, epsilon=0.3)

                        planning_time = time.time() - start_time
                        planning_times.append(planning_time)

                        next_state, reward, done, info = env.step(action)
                        episode_reward += reward
                        episode_length += 1

                        base_agent.store_experience(
                            state, action, reward, next_state, done
                        )
                        base_agent.update_q_function()

                        if episode_length % 5 == 0:
                            model_loss = base_agent.update_model()

                            if episode_length % 20 == 0:
                                accuracy = self._test_model_accuracy(
                                    model_ensemble, env
                                )
                                model_accuracy.append(accuracy)

                        state = next_state

                    episode_rewards.append(episode_reward)

                    if (episode + 1) % 50 == 0:
                        avg_reward = np.mean(episode_rewards[-20:])
                        avg_time = (
                            np.mean(planning_times[-100:]) if planning_times else 0
                        )
                        print(
                            f"    Episode {episode + 1}: Reward={avg_reward:.2f}, Planning Time={avg_time:.4f}s"
                        )

                planner_results.append(
                    {
                        "rewards": episode_rewards,
                        "planning_times": planning_times,
                        "model_accuracy": model_accuracy,
                        "final_performance": np.mean(episode_rewards[-20:]),
                    }
                )

            results[planner_name] = planner_results

        self.results = results
        return results

    def _test_model_accuracy(self, model_ensemble, env, num_tests=10):
        """Test how accurate the learned model is."""
        if len(model_ensemble.models) == 0:
            return 0.0

        accuracies = []

        for _ in range(num_tests):
            state = env.reset()
            action = np.random.randint(4)

            actual_next_state, actual_reward, _, _ = env.step(action)

            try:
                pred_next_state, pred_reward = model_ensemble.predict_mean(
                    torch.FloatTensor(state).to(device),
                    torch.LongTensor([action]).to(device),
                )

                state_error = torch.norm(
                    pred_next_state.cpu() - torch.FloatTensor(actual_next_state)
                ).item()
                reward_error = abs(pred_reward.cpu().item() - actual_reward)

                accuracy = 1.0 / (1.0 + state_error + reward_error)
                accuracies.append(accuracy)
            except:
                accuracies.append(0.0)  # Model failed

        return np.mean(accuracies) if accuracies else 0.0

    def visualize_planning_results(self):
        """Visualize planning algorithm comparison results."""
        if not self.results:
            print("❌ No results to visualize. Run experiment first.")
            return

        print("\n📊 Planning Algorithms Comparison Results")
        print("=" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Planning Algorithms Performance Analysis", fontsize=16)

        ax1 = axes[0, 0]
        colors = ["blue", "red", "green"]
        for i, (planner_name, planner_results) in enumerate(self.results.items()):
            all_rewards = [result["rewards"] for result in planner_results]
            min_length = min(len(rewards) for rewards in all_rewards)

            rewards_array = np.array([rewards[:min_length] for rewards in all_rewards])
            mean_rewards = np.mean(rewards_array, axis=0)
            std_rewards = np.std(rewards_array, axis=0)

            episodes = np.arange(min_length)
            ax1.plot(
                episodes, mean_rewards, label=planner_name, linewidth=2, color=colors[i]
            )
            ax1.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.3,
                color=colors[i],
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Reward")
        ax1.set_title("Learning Curves Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        planner_names = list(self.results.keys())
        planning_times = []
        time_stds = []

        for planner_name, planner_results in self.results.items():
            all_times = []
            for result in planner_results:
                if result["planning_times"]:
                    relevant_times = result["planning_times"][
                        len(result["planning_times"]) // 2 :
                    ]
                    all_times.extend(relevant_times)

            if all_times:
                planning_times.append(np.mean(all_times) * 1000)  # Convert to ms
                time_stds.append(np.std(all_times) * 1000)
            else:
                planning_times.append(0)
                time_stds.append(0)

        bars = ax2.bar(
            planner_names,
            planning_times,
            yerr=time_stds,
            capsize=5,
            color=["lightblue", "lightcoral", "lightgreen"],
        )
        ax2.set_ylabel("Average Planning Time (ms)")
        ax2.set_title("Computational Overhead")
        ax2.tick_params(axis="x", rotation=45)

        ax3 = axes[1, 0]
        final_performances = []
        perf_stds = []

        for planner_name, planner_results in self.results.items():
            performances = [result["final_performance"] for result in planner_results]
            final_performances.append(np.mean(performances))
            perf_stds.append(np.std(performances))

        bars = ax3.bar(
            planner_names,
            final_performances,
            yerr=perf_stds,
            capsize=5,
            color=["lightblue", "lightcoral", "lightgreen"],
        )
        ax3.set_ylabel("Final Average Reward")
        ax3.set_title("Final Performance")
        ax3.tick_params(axis="x", rotation=45)

        ax4 = axes[1, 1]
        for planner_name, planner_results in self.results.items():
            all_accuracies = []
            for result in planner_results:
                if result["model_accuracy"]:
                    all_accuracies.append(result["model_accuracy"])

            if all_accuracies:
                min_length = (
                    min(len(acc) for acc in all_accuracies) if all_accuracies else 0
                )
                if min_length > 0:
                    acc_array = np.array([acc[:min_length] for acc in all_accuracies])
                    mean_acc = np.mean(acc_array, axis=0)

                    time_steps = np.arange(len(mean_acc))
                    ax4.plot(time_steps, mean_acc, label=planner_name, linewidth=2)

        ax4.set_xlabel("Model Update Steps")
        ax4.set_ylabel("Model Accuracy")
        ax4.set_title("Model Learning Progress")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n📈 Planning Algorithms Summary:")
        for planner_name, planner_results in self.results.items():
            performances = [result["final_performance"] for result in planner_results]
            times = []
            for result in planner_results:
                if result["planning_times"]:
                    times.extend(result["planning_times"])

            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            mean_time = np.mean(times) * 1000 if times else 0  # ms

            print(f"\n{planner_name}:")
            print(f"  Final Performance: {mean_perf:.2f} ± {std_perf:.2f}")
            print(f"  Average Planning Time: {mean_time:.2f} ms")
            print(
                f"  Performance/Time Ratio: {mean_perf/max(mean_time/1000, 0.001):.1f}"
            )
