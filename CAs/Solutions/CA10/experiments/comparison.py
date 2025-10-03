import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import TabularModel, NeuralModel, ModelTrainer, device
from environments.environments import SimpleGridWorld
from agents.classical_planning import ModelBasedPlanner
from agents.dyna_q import DynaQAgent
from agents.mcts import MCTSAgent
from agents.mpc import MPCAgent


class ModelBasedComparisonFramework:
    """Framework for comparing different model-based RL approaches"""

    def __init__(self):
        self.results = {}
        self.environments = {}
        self.methods = {}

    def add_environment(self, name, env):
        """Add environment for testing"""
        self.environments[name] = env

    def add_method(self, name, method_class, **kwargs):
        """Add method to compare"""
        self.methods[name] = {"class": method_class, "kwargs": kwargs}

    def run_comparison(self, n_episodes=50, max_steps=200, n_runs=3):
        """Run comprehensive comparison"""
        print(f"\nRunning comprehensive comparison...")
        print(f"Episodes per run: {n_episodes}, Runs per method: {n_runs}")

        for env_name, env in self.environments.items():
            print(f"\n🌍 Environment: {env_name}")
            self.results[env_name] = {}

            if hasattr(env, "num_states"):
                tabular_model = TabularModel(env.num_states, env.num_actions)
                neural_model = NeuralModel(
                    env.num_states, env.num_actions, hidden_dim=64, ensemble_size=3
                )

                self._train_models(env, tabular_model, neural_model)

            for method_name, method_info in self.methods.items():
                print(f"  📊 Testing {method_name}...")

                method_results = []

                for run in range(n_runs):
                    kwargs = method_info["kwargs"].copy()

                    if "model" in kwargs:
                        if kwargs["model"] == "tabular":
                            kwargs["model"] = tabular_model
                        elif kwargs["model"] == "neural":
                            kwargs["model"] = neural_model

                    try:
                        agent = method_info["class"](**kwargs)

                        episode_rewards = []
                        episode_lengths = []

                        for episode in range(n_episodes):
                            reward, length = agent.train_episode(
                                env, max_steps=max_steps
                            )
                            episode_rewards.append(reward)
                            episode_lengths.append(length)

                        method_results.append(
                            {
                                "episode_rewards": episode_rewards,
                                "episode_lengths": episode_lengths,
                                "final_performance": np.mean(episode_rewards[-10:]),
                                "learning_efficiency": self._calculate_learning_efficiency(
                                    episode_rewards
                                ),
                                "statistics": (
                                    agent.get_statistics()
                                    if hasattr(agent, "get_statistics")
                                    else {}
                                ),
                            }
                        )

                    except Exception as e:
                        print(f"    ⚠️ Error with {method_name}: {str(e)}")
                        continue

                if method_results:
                    self.results[env_name][method_name] = self._aggregate_results(
                        method_results
                    )

                    avg_performance = self.results[env_name][method_name][
                        "avg_final_performance"
                    ]
                    std_performance = self.results[env_name][method_name][
                        "std_final_performance"
                    ]
                    print(
                        f"    ✅ Final Performance: {avg_performance:.3f} ± {std_performance:.3f}"
                    )

    def _train_models(self, env, tabular_model, neural_model, episodes=100):
        """Quick model training"""
        trainer = ModelTrainer(neural_model, lr=1e-3)

        experience_data = []
        for episode in range(episodes):
            state = env.reset()
            for step in range(50):
                action = np.random.randint(env.num_actions)
                next_state, reward, done = env.step(action)
                tabular_model.update(state, action, reward, next_state)
                experience_data.append((state, action, next_state, reward))
                if done:
                    break
                state = next_state

        if experience_data:
            states = np.array([exp[0] for exp in experience_data])
            actions = np.array([exp[1] for exp in experience_data])
            next_states = np.array([exp[2] for exp in experience_data])
            rewards = np.array([exp[3] for exp in experience_data])

            states_onehot = np.eye(env.num_states)[states]
            next_states_onehot = np.eye(env.num_states)[next_states]

            trainer.train_batch(
                (states_onehot, actions, next_states_onehot, rewards),
                epochs=20,
                batch_size=32,
            )

    def _calculate_learning_efficiency(self, rewards):
        """Calculate learning efficiency (area under learning curve)"""
        return np.sum(rewards) / len(rewards)

    def _aggregate_results(self, method_results):
        """Aggregate results across multiple runs"""
        final_performances = [r["final_performance"] for r in method_results]
        learning_efficiencies = [r["learning_efficiency"] for r in method_results]

        return {
            "avg_final_performance": np.mean(final_performances),
            "std_final_performance": np.std(final_performances),
            "avg_learning_efficiency": np.mean(learning_efficiencies),
            "std_learning_efficiency": np.std(learning_efficiencies),
            "all_results": method_results,
        }

    def visualize_results(self, save_path="visualizations"):
        """Create comprehensive visualization and save to folder"""
        import os

        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model-Based RL Comprehensive Comparison", fontsize=16)

        ax1 = axes[0, 0]
        for env_name, env_results in self.results.items():
            methods = list(env_results.keys())
            performances = [env_results[m]["avg_final_performance"] for m in methods]
            errors = [env_results[m]["std_final_performance"] for m in methods]

            x = np.arange(len(methods))
            ax1.bar(x, performances, yerr=errors, alpha=0.7, label=env_name, capsize=5)
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=45, ha="right")

        ax1.set_title("Final Performance Comparison")
        ax1.set_ylabel("Average Episode Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        for env_name, env_results in self.results.items():
            methods = list(env_results.keys())
            efficiencies = [env_results[m]["avg_learning_efficiency"] for m in methods]
            errors = [env_results[m]["std_learning_efficiency"] for m in methods]

            x = np.arange(len(methods))
            ax2.bar(x, efficiencies, yerr=errors, alpha=0.7, label=env_name, capsize=5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods, rotation=45, ha="right")

        ax2.set_title("Learning Efficiency")
        ax2.set_ylabel("Average Reward over Episodes")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if self.results:
            env_name = list(self.results.keys())[0]
            env_results = self.results[env_name]

            for method_name, method_data in env_results.items():
                if method_data["all_results"]:
                    all_rewards = [
                        r["episode_rewards"] for r in method_data["all_results"]
                    ]
                    if all_rewards:
                        avg_rewards = np.mean(all_rewards, axis=0)
                        smoothed = pd.Series(avg_rewards).rolling(window=5).mean()
                        ax3.plot(smoothed, label=method_name, linewidth=2)

            ax3.set_title(f"Learning Curves - {env_name}")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Episode Reward (Smoothed)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.text(
            0.5,
            0.5,
            "Method Characteristics:\\n\\n"
            "• Sample Efficiency\\n"
            "• Computational Cost\\n"
            "• Adaptability\\n"
            "• Theoretical Guarantees\\n"
            "• Implementation Complexity",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )
        ax4.set_title("Key Method Properties")
        ax4.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{save_path}/comprehensive_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        self._save_individual_plots(save_path)

    def _save_individual_plots(self, save_path):
        """Save individual comparison plots"""
        if not self.results:
            return

        env_name = list(self.results.keys())[0]
        env_results = self.results[env_name]

        plt.figure(figsize=(12, 8))
        methods = list(env_results.keys())
        performances = [env_results[m]["avg_final_performance"] for m in methods]
        errors = [env_results[m]["std_final_performance"] for m in methods]

        bars = plt.bar(
            methods,
            performances,
            yerr=errors,
            capsize=5,
            color=["skyblue", "lightgreen", "lightcoral", "gold", "violet"],
        )
        plt.title(f"Model-Based RL Performance Comparison - {env_name}")
        plt.ylabel("Average Final Episode Reward")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        for bar, perf in zip(bars, performances):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                ".3f",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            f"{save_path}/performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        plt.figure(figsize=(12, 8))
        efficiencies = [env_results[m]["avg_learning_efficiency"] for m in methods]
        errors = [env_results[m]["std_learning_efficiency"] for m in methods]

        bars = plt.bar(
            methods,
            efficiencies,
            yerr=errors,
            capsize=5,
            color=["skyblue", "lightgreen", "lightcoral", "gold", "violet"],
        )
        plt.title(f"Learning Efficiency Comparison - {env_name}")
        plt.ylabel("Average Reward over Episodes")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        for bar, eff in zip(bars, efficiencies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                ".3f",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            f"{save_path}/learning_efficiency.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        plt.figure(figsize=(14, 8))
        for method_name, method_data in env_results.items():
            if method_data["all_results"]:
                all_rewards = [r["episode_rewards"] for r in method_data["all_results"]]
                if all_rewards:
                    avg_rewards = np.mean(all_rewards, axis=0)
                    smoothed = pd.Series(avg_rewards).rolling(window=5).mean()
                    plt.plot(smoothed, label=method_name, linewidth=2)

        plt.title(f"Learning Curves - {env_name}")
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/learning_curves.png", dpi=300, bbox_inches="tight")
        plt.show()

    def print_summary(self):
        """Print comprehensive summary"""
        print(f"\n📋 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)

        for env_name, env_results in self.results.items():
            print(f"\n🌍 Environment: {env_name}")
            print("-" * 40)

            sorted_methods = sorted(
                env_results.items(),
                key=lambda x: x[1]["avg_final_performance"],
                reverse=True,
            )

            print("Performance Ranking:")
            for i, (method_name, data) in enumerate(sorted_methods, 1):
                perf = data["avg_final_performance"]
                std = data["std_final_performance"]
                eff = data["avg_learning_efficiency"]
                print(
                    f"  {i}. {method_name}: {perf:.3f} ± {std:.3f} "
                    f"(efficiency: {eff:.3f})"
                )


def demonstrate_comparison():
    """Comprehensive Model-Based RL Comparison"""

    print("Comprehensive Model-Based Reinforcement Learning Analysis")
    print("=" * 60)

    framework = ModelBasedComparisonFramework()

    framework.add_environment("GridWorld-5x5", SimpleGridWorld(size=5))

    framework.add_method(
        "Q-Learning",
        lambda **kwargs: DynaQAgent(25, 4, planning_steps=0),
        num_states=25,
        num_actions=4,
    )

    framework.add_method(
        "Dyna-Q(5)",
        lambda **kwargs: DynaQAgent(25, 4, planning_steps=5),
        num_states=25,
        num_actions=4,
    )

    framework.add_method(
        "Dyna-Q(20)",
        lambda **kwargs: DynaQAgent(25, 4, planning_steps=20),
        num_states=25,
        num_actions=4,
    )

    framework.add_method(
        "MCTS",
        lambda **kwargs: MCTSAgent(kwargs["model"], 25, 4, num_simulations=50),
        model="tabular",
    )

    framework.run_comparison(n_episodes=30, n_runs=2)

    framework.visualize_results(save_path="visualizations")
    framework.print_summary()

    print(f"\n🎯 FINAL CONCLUSIONS: Model-Based Reinforcement Learning")
    print("=" * 60)

    print(f"\n📊 Key Findings:")
    print(
        "1. Sample Efficiency: Model-based methods generally require fewer environment interactions"
    )
    print("2. Planning Benefits: More planning steps typically improve performance")
    print("3. Model Quality: Better models lead to better planning performance")
    print(
        "4. Computational Trade-offs: Planning methods trade computation for sample efficiency"
    )
    print("5. Adaptability: Some methods (Dyna-Q+) handle environment changes better")

    print(f"\n🔬 Method Characteristics:")
    print("• Tabular Models: Simple, exact, limited to discrete spaces")
    print("• Neural Models: Flexible, scalable, but require careful training")
    print("• Dyna-Q: Simple integration of learning and planning")
    print("• MCTS: Sophisticated tree search, good for discrete actions")
    print("• MPC: Principled control theory approach, handles constraints")

    print(f"\n💡 Practical Recommendations:")
    print("1. Use model-based methods when sample efficiency is critical")
    print("2. Choose tabular models for small discrete environments")
    print("3. Use neural models for high-dimensional or continuous spaces")
    print("4. Apply Dyna-Q for balanced learning and planning")
    print("5. Use MCTS for complex decision trees")
    print("6. Apply MPC when constraints are important")

    print(f"\n🚀 Future Directions:")
    print("• Uncertainty-aware planning")
    print("• Hierarchical model-based RL")
    print("• Meta-learning for quick model adaptation")
    print("• Differentiable planning modules")
    print("• Hybrid model-free and model-based methods")

    print(f"\n✅ MODEL-BASED REINFORCEMENT LEARNING COMPLETE!")
    print("🎓 You now have a comprehensive understanding of:")
    print("   • Theoretical foundations and mathematical formulations")
    print("   • Environment model learning (tabular and neural)")
    print("   • Classical planning with learned models")
    print("   • Dyna-Q algorithm for integrated learning and planning")
    print("   • Monte Carlo Tree Search (MCTS) for sophisticated planning")
    print("   • Model Predictive Control (MPC) for constrained optimization")
    print("   • Modern approaches and state-of-the-art methods")
    print("   • Comparative analysis and practical guidelines")

    print(f"\n🌟 Congratulations on completing this comprehensive study!")
    print("📚 Continue exploring advanced topics in model-based RL!")


if __name__ == "__main__":
    demonstrate_comparison()
