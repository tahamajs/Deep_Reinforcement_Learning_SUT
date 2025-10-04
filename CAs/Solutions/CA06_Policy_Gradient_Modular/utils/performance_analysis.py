import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .setup import device
import gymnasium as gym
from collections import defaultdict
import time
import warnings

warnings.filterwarnings("ignore")


class PolicyEvaluator:
    """
    Comprehensive policy evaluation framework
    """

    def __init__(self, env_name="CartPole-v1", num_episodes=100):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.env = gym.make(env_name)

    def evaluate_policy(self, agent, deterministic=True, render=False):
        """Evaluate a policy's performance"""
        episode_rewards = []
        episode_lengths = []
        successes = []

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                if hasattr(agent, "select_action"):
                    action = agent.select_action(state)
                else:

                    action = agent(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                episode_length += 1
                state = next_state
                done = terminated or truncated

                if render and episode == 0:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if self.env_name == "CartPole-v1":
                success = episode_reward >= 195
            elif self.env_name == "Pendulum-v1":
                success = episode_reward >= -200
            else:
                success = episode_length >= 100

            successes.append(success)

        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "median_reward": np.median(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": np.mean(successes),
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "successes": successes,
        }

        return results

    def compare_policies(self, agents_dict, names=None):
        """Compare multiple policies"""
        if names is None:
            names = list(agents_dict.keys())

        results = {}

        print(f"=== Policy Comparison on {self.env_name} ===")

        for name, agent in agents_dict.items():
            print(f"\nEvaluating {name}...")
            start_time = time.time()
            results[name] = self.evaluate_policy(agent, deterministic=True)
            eval_time = time.time() - start_time

            print(f"Mean Reward: {results[name]['mean_reward']:.2f}")
            print(f"Std Reward: {results[name]['std_reward']:.2f}")
            print(f"Success Rate: {results[name]['success_rate']:.1%}")
            print(f"Evaluation Time: {eval_time:.3f}s")

        return results


class PerformanceAnalyzer:
    """
    Advanced performance analysis tools
    """

    def __init__(self):
        self.analysis_results = {}

    def analyze_learning_curves(self, training_data, window=20):
        """Analyze learning curve characteristics"""
        analysis = {}

        for name, data in training_data.items():
            rewards = np.array(data["rewards"])

            smoothed = pd.Series(rewards).rolling(window=window).mean().dropna()

            final_performance = np.mean(rewards[-window:])
            peak_performance = np.max(smoothed)
            convergence_episode = np.where(smoothed >= 0.9 * peak_performance)[0]

            if len(convergence_episode) > 0:
                convergence_episode = convergence_episode[0] + window
            else:
                convergence_episode = len(rewards)

            final_std = np.std(rewards[-window:])
            variability = np.std(smoothed)

            early_performance = np.mean(rewards[:window])
            improvement_rate = (final_performance - early_performance) / len(rewards)

            analysis[name] = {
                "final_performance": final_performance,
                "peak_performance": peak_performance,
                "convergence_episode": convergence_episode,
                "final_stability": final_std,
                "variability": variability,
                "improvement_rate": improvement_rate,
                "smoothed_curve": smoothed.values,
            }

        return analysis

    def statistical_comparison(self, results_dict, metric="mean_reward"):
        """Perform statistical comparison between methods"""
        methods = list(results_dict.keys())
        values = [results_dict[method][metric] for method in methods]

        stats = {
            "best_method": methods[np.argmax(values)],
            "best_score": np.max(values),
            "method_ranks": sorted(
                zip(methods, values), key=lambda x: x[1], reverse=True
            ),
        }

        for method in methods:
            data = results_dict[method]["rewards"]
            mean = np.mean(data)
            std = np.std(data)
            n = len(data)

            ci_lower = mean - 1.96 * std / np.sqrt(n)
            ci_upper = mean + 1.96 * std / np.sqrt(n)

            stats[f"{method}_ci"] = (ci_lower, ci_upper)

        return stats

    def sample_efficiency_analysis(self, training_data):
        """Analyze sample efficiency of different methods"""
        efficiency = {}

        for name, data in training_data.items():
            rewards = np.array(data["rewards"])

            thresholds = [0.5, 0.7, 0.9]
            threshold_episodes = {}

            for threshold in thresholds:
                target_reward = threshold * np.max(rewards)
                reached_episodes = np.where(rewards >= target_reward)[0]

                if len(reached_episodes) > 0:
                    threshold_episodes[f"to_{int(threshold*100)}%"] = reached_episodes[
                        0
                    ]
                else:
                    threshold_episodes[f"to_{int(threshold*100)}%"] = len(rewards)

            efficiency[name] = threshold_episodes

        return efficiency


class AblationStudy:
    """
    Framework for ablation studies
    """

    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}

    def run_ablation(self, parameter, values, agent_class, env_name="CartPole-v1"):
        """Run ablation study for a parameter"""
        print(f"=== Ablation Study: {parameter} ===")

        for value in values:
            print(f"\nTesting {parameter} = {value}")

            config = self.base_config.copy()
            config[parameter] = value

            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            agent = agent_class(state_dim, action_dim, **config)

            rewards = []
            for episode in range(100):
                episode_reward, _ = agent.train_episode(env)
                rewards.append(episode_reward)

            final_score = np.mean(rewards[-20:])

            self.results[f"{parameter}_{value}"] = {
                "value": value,
                "final_score": final_score,
                "rewards": rewards,
            }

            print(".2f")

        env.close()
        return self.results


class RobustnessTester:
    """
    Test policy robustness under different conditions
    """

    def __init__(self, env_name="CartPole-v1"):
        self.env_name = env_name

    def test_parameter_sensitivity(
        self, agent, param_name, param_values, num_episodes=50
    ):
        """Test sensitivity to environment parameters"""
        results = {}

        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")

            if param_name == "gravity":
                env = gym.make(self.env_name, g=param_value)
            elif param_name == "mass":
                env = gym.make(self.env_name, masscart=param_value)
            elif param_name == "length":
                env = gym.make(self.env_name, length=param_value)
            else:
                env = gym.make(self.env_name)

            evaluator = PolicyEvaluator(env.env_name, num_episodes)
            results[param_value] = evaluator.evaluate_policy(agent)

            env.close()

        return results

    def test_noise_robustness(self, agent, noise_levels, num_episodes=50):
        """Test robustness to action noise"""
        results = {}

        for noise in noise_levels:
            print(f"Testing noise level = {noise}")

            class NoisyEnv(gym.Wrapper):
                def __init__(self, env, noise_level):
                    super().__init__(env)
                    self.noise_level = noise_level

                def step(self, action):

                    if isinstance(action, np.ndarray):
                        noisy_action = action + np.random.normal(
                            0, self.noise_level, size=action.shape
                        )

                        noisy_action = np.clip(
                            noisy_action, self.action_space.low, self.action_space.high
                        )
                    else:
                        noisy_action = action + np.random.normal(0, self.noise_level)
                        noisy_action = np.clip(
                            noisy_action, self.action_space.low, self.action_space.high
                        )

                    return self.env.step(noisy_action)

            env = NoisyEnv(gym.make(self.env_name), noise)
            evaluator = PolicyEvaluator(env.env_name, num_episodes)
            results[noise] = evaluator.evaluate_policy(agent)

            env.close()

        return results


def create_comprehensive_report(training_results, evaluation_results):
    """Create comprehensive performance report"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)

    analyzer = PerformanceAnalyzer()
    learning_analysis = analyzer.analyze_learning_curves(training_results)
    stats_comparison = analyzer.statistical_comparison(evaluation_results)

    print("\n📈 LEARNING CURVE ANALYSIS:")
    for method, analysis in learning_analysis.items():
        print(f"\n{method}:")
        print(f"  Final Performance: {analysis['final_performance']:.2f}")
        print(f"  Peak Performance: {analysis['peak_performance']:.2f}")
        print(f"  Convergence Episode: {analysis['convergence_episode']}")
        print(f"  Final Stability: {analysis['final_stability']:.4f}")
        print(f"  Variability: {analysis['variability']:.4f}")

    print("\n🏆 STATISTICAL COMPARISON:")
    print(f"Best Method: {stats_comparison['best_method']}")
    print(f"Best Score: {stats_comparison['best_score']:.2f}")

    print("\n📊 CONFIDENCE INTERVALS (95%):")
    for method in evaluation_results.keys():
        ci_lower, ci_upper = stats_comparison[f"{method}_ci"]
        print(f"{method:15} | CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    efficiency = analyzer.sample_efficiency_analysis(training_results)
    print("\n⚡ SAMPLE EFFICIENCY:")
    for method, thresholds in efficiency.items():
        print(f"\n{method}:")
        for threshold, episodes in thresholds.items():
            print(f"  {threshold}: {episodes} episodes")

    return {
        "learning_analysis": learning_analysis,
        "stats_comparison": stats_comparison,
        "sample_efficiency": efficiency,
    }


def visualize_performance_comparison(results_dict, title="Performance Comparison"):
    """Create comprehensive visualization of results"""
    methods = list(results_dict.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        data = results_dict[method]
        color = colors[i]

        axes[0, 0].plot(data["rewards"], color=color, alpha=0.7, label=method)
        axes[0, 0].plot(
            pd.Series(data["rewards"]).rolling(window=20).mean(),
            color=color,
            linestyle="--",
            alpha=0.9,
        )

        axes[0, 1].hist(data["rewards"], bins=20, alpha=0.7, color=color, label=method)

        axes[0, 2].plot(np.cumsum(data["rewards"]), color=color, label=method)

        metrics = ["mean_reward", "std_reward", "success_rate"]
        metric_names = ["Mean Reward", "Std Reward", "Success Rate"]

        for j, (metric, name) in enumerate(zip(metrics, metric_names)):
            if metric in data:
                axes[1, j].bar(method, data[metric], color=color, alpha=0.7)

    axes[0, 0].set_title("Learning Curves")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Reward Distributions")
    axes[0, 1].set_xlabel("Episode Reward")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    axes[0, 2].set_title("Cumulative Rewards")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Cumulative Reward")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    for j, name in enumerate(["Mean Reward", "Std Reward", "Success Rate"]):
        axes[1, j].set_title(name)
        axes[1, j].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def demonstrate_performance_analysis():
    """Demonstrate performance analysis tools"""
    print("📊 Performance Analysis Demonstration")

    np.random.seed(42)

    mock_training_data = {
        "REINFORCE": {
            "rewards": np.cumsum(np.random.normal(1, 2, 300))
            + np.random.normal(0, 5, 300)
        },
        "Actor-Critic": {
            "rewards": np.cumsum(np.random.normal(1.5, 1.5, 300))
            + np.random.normal(0, 3, 300)
        },
        "PPO": {
            "rewards": np.cumsum(np.random.normal(2, 1, 300))
            + np.random.normal(0, 2, 300)
        },
    }

    mock_evaluation_data = {
        "REINFORCE": {
            "mean_reward": 150.5,
            "std_reward": 25.3,
            "success_rate": 0.75,
            "rewards": np.random.normal(150.5, 25.3, 100),
        },
        "Actor-Critic": {
            "mean_reward": 180.2,
            "std_reward": 15.7,
            "success_rate": 0.88,
            "rewards": np.random.normal(180.2, 15.7, 100),
        },
        "PPO": {
            "mean_reward": 195.8,
            "std_reward": 8.9,
            "success_rate": 0.95,
            "rewards": np.random.normal(195.8, 8.9, 100),
        },
    }

    report = create_comprehensive_report(mock_training_data, mock_evaluation_data)

    visualize_performance_comparison(mock_evaluation_data)

    return report


def generate_comprehensive_report():
    """Generate comprehensive report for CA06"""
    print("Generating comprehensive report...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs("results", exist_ok=True)
    
    # Run the demonstration
    report = demonstrate_performance_analysis()
    
    # Save report to file
    with open("results/comprehensive_report.txt", "w") as f:
        f.write("CA06 Policy Gradient Methods - Comprehensive Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(str(report))
    
    print("✅ Comprehensive report saved to results/comprehensive_report.txt")
    return report


if __name__ == "__main__":
    demonstrate_performance_analysis()
