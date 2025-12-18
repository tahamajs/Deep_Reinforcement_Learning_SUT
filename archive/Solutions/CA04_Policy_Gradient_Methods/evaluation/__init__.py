"""
Evaluation utilities for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import numpy as np
import torch
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os


class PolicyGradientEvaluator:
    """Comprehensive evaluator for policy gradient methods"""

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize evaluator

        Args:
            env_name: Environment name
        """
        self.env_name = env_name
        self.evaluation_results = {}
        self.metrics_history = []

    def evaluate_agent(
        self, agent, env, num_episodes: int = 100, render: bool = False
    ) -> Dict[str, Any]:
        """Evaluate agent performance

        Args:
            agent: Trained agent
            env: Environment
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes

        Returns:
            Evaluation results dictionary
        """
        scores = []
        episode_lengths = []
        success_rate = 0

        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]

            episode_score = 0
            episode_length = 0
            done = False
            truncated = False

            while not (done or truncated):
                if hasattr(agent, "get_action"):
                    action, _ = agent.get_action(state)
                elif hasattr(agent, "get_action_and_value"):
                    action, _, _ = agent.get_action_and_value(state)
                else:
                    action = env.action_space.sample()

                next_state, reward, done, truncated, _ = env.step(action)
                episode_score += reward
                episode_length += 1
                state = next_state

                if render and episode % 10 == 0:
                    env.render()

            scores.append(episode_score)
            episode_lengths.append(episode_length)

            # تعریف موفقیت برای CartPole (امتیاز > 195)
            if episode_score >= 195:
                success_rate += 1

        success_rate = success_rate / num_episodes

        results = {
            "scores": scores,
            "episode_lengths": episode_lengths,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "success_rate": success_rate,
            "mean_length": np.mean(episode_lengths),
            "evaluation_episodes": num_episodes,
            "timestamp": datetime.now().isoformat(),
        }

        return results

    def compare_agents(
        self, agents: Dict[str, Any], env, num_episodes: int = 100
    ) -> Dict[str, Dict]:
        """Compare multiple agents

        Args:
            agents: Dictionary of agent names to agents
            env: Environment
            num_episodes: Episodes per agent

        Returns:
            Comparison results
        """
        comparison_results = {}

        for agent_name, agent in agents.items():
            print(f"ارزیابی {agent_name}...")
            results = self.evaluate_agent(agent, env, num_episodes)
            comparison_results[agent_name] = results

        self.evaluation_results["comparison"] = comparison_results
        return comparison_results

    def statistical_significance_test(
        self, results1: Dict, results2: Dict, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Perform statistical significance test

        Args:
            results1: First agent results
            results2: Second agent results
            alpha: Significance level

        Returns:
            Statistical test results
        """
        from scipy import stats

        scores1 = results1["scores"]
        scores2 = results2["scores"]

        # t-test
        t_stat, p_value = stats.ttest_ind(scores1, scores2)

        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(
            scores1, scores2, alternative="two-sided"
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(scores1) - 1) * np.var(scores1, ddof=1)
                + (len(scores2) - 1) * np.var(scores2, ddof=1)
            )
            / (len(scores1) + len(scores2) - 2)
        )
        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std

        test_results = {
            "t_statistic": t_stat,
            "t_p_value": p_value,
            "t_significant": p_value < alpha,
            "u_statistic": u_stat,
            "u_p_value": u_p_value,
            "u_significant": u_p_value < alpha,
            "cohens_d": cohens_d,
            "effect_size": (
                "small"
                if abs(cohens_d) < 0.5
                else "medium" if abs(cohens_d) < 0.8 else "large"
            ),
            "alpha": alpha,
        }

        return test_results

    def learning_curve_analysis(
        self, scores: List[float], window: int = 50
    ) -> Dict[str, Any]:
        """Analyze learning curve properties

        Args:
            scores: List of episode scores
            window: Window size for moving average

        Returns:
            Learning curve analysis
        """
        if len(scores) < window:
            return {"error": "Not enough data for analysis"}

        # Moving average
        moving_avg = [
            np.mean(scores[i - window : i]) for i in range(window, len(scores))
        ]

        # Convergence analysis
        final_performance = np.mean(scores[-window:])
        max_performance = np.max(moving_avg)
        convergence_threshold = 0.95 * max_performance

        convergence_episode = None
        for i, avg in enumerate(moving_avg):
            if avg >= convergence_threshold:
                convergence_episode = i + window
                break

        # Stability analysis
        final_window_scores = scores[-window:]
        stability = 1.0 - (np.std(final_window_scores) / np.mean(final_window_scores))

        # Sample efficiency
        sample_efficiency = (
            convergence_episode / len(scores) if convergence_episode else 1.0
        )

        analysis = {
            "final_performance": final_performance,
            "max_performance": max_performance,
            "convergence_episode": convergence_episode,
            "convergence_threshold": convergence_threshold,
            "stability": stability,
            "sample_efficiency": sample_efficiency,
            "total_episodes": len(scores),
            "improvement": final_performance - np.mean(scores[:window]),
        }

        return analysis

    def save_results(self, filename: str, results: Dict[str, Any]):
        """Save evaluation results

        Args:
            filename: Output filename
            results: Results to save
        """
        os.makedirs("evaluation/results", exist_ok=True)

        with open(f"evaluation/results/{filename}", "wb") as f:
            pickle.dump(results, f)

        # Also save as JSON for human readability
        json_filename = filename.replace(".pkl", ".json")
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                json_results[key] = [
                    float(x) if isinstance(x, (int, float, np.number)) else str(x)
                    for x in value
                ]
            elif isinstance(value, dict):
                json_results[key] = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                    for k, v in value.items()
                }
            else:
                json_results[key] = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else str(value)
                )

        with open(f"evaluation/results/{json_filename}", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load evaluation results

        Args:
            filename: Input filename

        Returns:
            Loaded results
        """
        with open(f"evaluation/results/{filename}", "rb") as f:
            return pickle.load(f)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report

        Args:
            results: Evaluation results

        Returns:
            Formatted report string
        """
        report = f"""
گزارش ارزیابی Policy Gradient Methods
=====================================
تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
محیط: {self.env_name}

"""

        if "comparison" in results:
            report += "نتایج مقایسه الگوریتم‌ها:\n"
            report += "-" * 40 + "\n"

            for agent_name, agent_results in results["comparison"].items():
                report += f"\n{agent_name.upper()}:\n"
                report += f"  میانگین امتیاز: {agent_results['mean_score']:.2f} ± {agent_results['std_score']:.2f}\n"
                report += f"  بهترین امتیاز: {agent_results['max_score']:.2f}\n"
                report += f"  نرخ موفقیت: {agent_results['success_rate']:.2%}\n"
                report += f"  میانگین طول اپیزود: {agent_results['mean_length']:.2f}\n"

        return report


class ModelSaver:
    """Save and load trained models"""

    def __init__(self, base_path: str = "models/saved_models"):
        """Initialize model saver

        Args:
            base_path: Base path for saving models
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_model(self, model, filename: str, metadata: Dict[str, Any] = None):
        """Save PyTorch model

        Args:
            model: PyTorch model
            filename: Output filename
            metadata: Additional metadata
        """
        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        filepath = os.path.join(self.base_path, filename)
        torch.save(save_dict, filepath)
        print(f"مدل ذخیره شد: {filepath}")

    def load_model(self, filename: str, model_class=None):
        """Load PyTorch model

        Args:
            filename: Input filename
            model_class: Model class for loading

        Returns:
            Loaded model
        """
        filepath = os.path.join(self.base_path, filename)
        save_dict = torch.load(filepath, map_location="cpu")

        if model_class is None:
            raise ValueError("model_class must be provided for loading")

        model = model_class(**save_dict["metadata"].get("init_params", {}))
        model.load_state_dict(save_dict["model_state_dict"])

        print(f"مدل بارگذاری شد: {filepath}")
        return model

    def list_models(self) -> List[str]:
        """List available models

        Returns:
            List of model filenames
        """
        return [f for f in os.listdir(self.base_path) if f.endswith(".pth")]


def create_evaluation_summary(results: Dict[str, Any]) -> str:
    """Create summary of evaluation results

    Args:
        results: Evaluation results

    Returns:
        Summary string
    """
    summary = "خلاصه ارزیابی:\n"
    summary += "=" * 30 + "\n"

    if "comparison" in results:
        best_agent = None
        best_score = -float("inf")

        for agent_name, agent_results in results["comparison"].items():
            score = agent_results["mean_score"]
            if score > best_score:
                best_score = score
                best_agent = agent_name

        summary += f"بهترین الگوریتم: {best_agent}\n"
        summary += f"بهترین امتیاز: {best_score:.2f}\n"

    return summary


