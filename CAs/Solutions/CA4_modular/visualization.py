"""
Visualization utilities for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch


class PolicyVisualizer:
    """Visualize different policy representations"""

    def __init__(self, n_states: int = 4, n_actions: int = 2):
        """Initialize policy visualizer

        Args:
            n_states: Number of states
            n_actions: Number of actions
        """
        self.n_states = n_states
        self.n_actions = n_actions

    def softmax_policy(self, preferences: np.ndarray) -> np.ndarray:
        """Softmax policy for discrete actions

        Args:
            preferences: Action preferences

        Returns:
            Action probabilities
        """
        exp_prefs = np.exp(preferences - np.max(preferences))  # Numerical stability
        return exp_prefs / np.sum(exp_prefs)

    def gaussian_policy(self, mu: float, sigma: float, action: float) -> float:
        """Gaussian policy for continuous actions

        Args:
            mu: Mean
            sigma: Standard deviation
            action: Action value

        Returns:
            Probability density
        """
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((action - mu) / sigma) ** 2)

    def visualize_policies(self):
        """Compare different policy types"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        states = range(self.n_states)
        deterministic_probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        stochastic_probs = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]])

        x = np.arange(len(states))
        width = 0.35

        axes[0,0].bar(x - width/2, deterministic_probs[:, 0], width,
                     label='Action 0', alpha=0.8, color='skyblue')
        axes[0,0].bar(x + width/2, deterministic_probs[:, 1], width,
                     label='Action 1', alpha=0.8, color='lightcoral')
        axes[0,0].set_title('Deterministic Policy')
        axes[0,0].set_xlabel('State')
        axes[0,0].set_ylabel('Action Probability')
        axes[0,0].legend()

        axes[0,1].bar(x - width/2, stochastic_probs[:, 0], width,
                     label='Action 0', alpha=0.8, color='skyblue')
        axes[0,1].bar(x + width/2, stochastic_probs[:, 1], width,
                     label='Action 1', alpha=0.8, color='lightcoral')
        axes[0,1].set_title('Stochastic Policy')
        axes[0,1].set_xlabel('State')
        axes[0,1].set_ylabel('Action Probability')
        axes[0,1].legend()

        preferences = np.array([2.0, 1.0, 0.5])
        temperatures = [0.1, 1.0, 10.0]

        for i, temp in enumerate(temperatures):
            probs = self.softmax_policy(preferences / temp)
            axes[1,0].plot(preferences, probs, 'o-',
                          label=f'Temperature = {temp}', linewidth=2, markersize=8)

        axes[1,0].set_title('Softmax Policy with Different Temperatures')
        axes[1,0].set_xlabel('Action Preferences')
        axes[1,0].set_ylabel('Action Probability')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        actions = np.linspace(-3, 3, 100)
        mu_values = [0.0, 1.0, -0.5]
        sigma_values = [0.5, 1.0, 1.5]

        for mu, sigma in zip(mu_values, sigma_values):
            probs = [self.gaussian_policy(mu, sigma, a) for a in actions]
            axes[1,1].plot(actions, probs, linewidth=2,
                          label=f'μ={mu}, σ={sigma}')

        axes[1,1].set_title('Gaussian Policy for Continuous Actions')
        axes[1,1].set_xlabel('Action Value')
        axes[1,1].set_ylabel('Probability Density')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class PolicyGradientMathVisualizer:
    """Demonstrate policy gradient mathematical concepts"""

    def __init__(self):
        """Initialize math visualizer"""
        self.n_states = 3
        self.n_actions = 2

    def softmax_policy_gradient(self, preferences: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient of log softmax policy

        Args:
            preferences: Action preferences
            action: Selected action

        Returns:
            Tuple of (probabilities, gradients)
        """
        exp_prefs = np.exp(preferences - np.max(preferences))
        probs = exp_prefs / np.sum(exp_prefs)

        grad_log_policy = np.zeros_like(preferences)
        grad_log_policy[action] = 1.0
        grad_log_policy -= probs

        return probs, grad_log_policy

    def demonstrate_score_function(self):
        """Visualize score function properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        preferences = np.array([1.0, 2.0])
        actions = [0, 1]

        pref_range = np.linspace(-2, 4, 100)

        for action in actions:
            scores = []
            for pref in pref_range:
                current_prefs = preferences.copy()
                current_prefs[action] = pref
                _, grad = self.softmax_policy_gradient(current_prefs, action)
                scores.append(grad[action])

            axes[0,0].plot(pref_range, scores, linewidth=2,
                          label=f'Action {action}')

        axes[0,0].set_title('Score Function: ∇_θ log π(a|s,θ)')
        axes[0,0].set_xlabel('Action Preference θ_a')
        axes[0,0].set_ylabel('Score')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for action in actions:
            probs = []
            for pref in pref_range:
                current_prefs = preferences.copy()
                current_prefs[action] = pref
                prob, _ = self.softmax_policy_gradient(current_prefs, action)
                probs.append(prob[action])

            axes[0,1].plot(pref_range, probs, linewidth=2,
                          label=f'π(a={action}|s,θ)')

        axes[0,1].set_title('Policy Probabilities')
        axes[0,1].set_xlabel('Action Preference θ_a')
        axes[0,1].set_ylabel('Probability')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        returns = np.random.normal(10, 5, 1000)  # Sample returns
        baseline_values = np.linspace(5, 15, 50)
        variances = []

        for baseline in baseline_values:
            adjusted_returns = returns - baseline
            variances.append(np.var(adjusted_returns))

        axes[1,0].plot(baseline_values, variances, linewidth=2, color='red')
        optimal_baseline = np.mean(returns)
        axes[1,0].axvline(x=optimal_baseline, color='blue', linestyle='--',
                         label=f'Optimal baseline = {optimal_baseline:.2f}')
        axes[1,0].set_title('Variance Reduction with Baseline')
        axes[1,0].set_xlabel('Baseline Value')
        axes[1,0].set_ylabel('Variance')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        n_episodes = 500
        true_return = 10.0
        noise_std = 3.0

        gradients_no_baseline = []
        returns_sample = np.random.normal(true_return, noise_std, n_episodes)

        gradients_with_baseline = []
        baseline = np.mean(returns_sample)

        for episode in range(n_episodes):
            grad_no_baseline = returns_sample[episode]  # G_t
            grad_with_baseline = returns_sample[episode] - baseline  # G_t - b

            gradients_no_baseline.append(grad_no_baseline)
            gradients_with_baseline.append(grad_with_baseline)

        window = 50
        var_no_baseline = []
        var_with_baseline = []

        for i in range(window, n_episodes):
            var_no_baseline.append(np.var(gradients_no_baseline[i-window:i]))
            var_with_baseline.append(np.var(gradients_with_baseline[i-window:i]))

        episodes = range(window, n_episodes)
        axes[1,1].plot(episodes, var_no_baseline, label='Without Baseline',
                      linewidth=2, alpha=0.8)
        axes[1,1].plot(episodes, var_with_baseline, label='With Baseline',
                      linewidth=2, alpha=0.8)
        axes[1,1].set_title('Gradient Variance Over Training')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Gradient Variance')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class TrainingVisualizer:
    """Visualize training progress and results"""

    def __init__(self):
        """Initialize training visualizer"""
        pass

    def plot_learning_curves(self, scores: List[float], title: str = "Learning Curve",
                           window: int = 20, show_std: bool = False):
        """Plot learning curve with moving average

        Args:
            scores: List of episode scores
            title: Plot title
            window: Window size for moving average
            show_std: Whether to show standard deviation
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(scores, alpha=0.6, color='blue', label='Episode Rewards')

        if len(scores) >= window:
            moving_avg = [np.mean(scores[i-window:i]) for i in range(window, len(scores))]
            ax.plot(range(window, len(scores)), moving_avg,
                   color='red', linewidth=2, label=f'{window}-Episode Average')

            if show_std:
                moving_std = [np.std(scores[i-window:i]) for i in range(window, len(scores))]
                ax.fill_between(range(window, len(scores)),
                               np.array(moving_avg) - np.array(moving_std),
                               np.array(moving_avg) + np.array(moving_std),
                               alpha=0.2, color='red')

        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.show()

    def plot_multiple_curves(self, results: Dict[str, List[float]], title: str = "Comparison",
                           window: int = 20):
        """Plot multiple learning curves for comparison

        Args:
            results: Dictionary of algorithm names to scores
            title: Plot title
            window: Window size for moving average
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        episodes = range(len(list(results.values())[0]))

        # Learning curves
        for name, scores in results.items():
            axes[0,0].plot(scores, alpha=0.6, label=name)

        for name, scores in results.items():
            if len(scores) >= window:
                moving_avg = [np.mean(scores[i-window:i]) for i in range(window, len(scores))]
                axes[0,0].plot(range(window, len(scores)), moving_avg,
                              linewidth=2, label=f'{name} (avg)')

        axes[0,0].set_title('Learning Curves')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Performance distribution
        data = list(results.values())
        labels = list(results.keys())
        axes[0,1].boxplot(data, labels=labels)
        axes[0,1].set_title('Performance Distribution')
        axes[0,1].set_ylabel('Episode Reward')
        axes[0,1].grid(True, alpha=0.3)

        # Learning stability (variance)
        window_size = 50
        for name, scores in results.items():
            if len(scores) >= window_size:
                variances = []
                for i in range(window_size, len(scores)):
                    var = np.var(scores[i-window_size:i])
                    variances.append(var)

                axes[1,0].plot(range(window_size, len(scores)), variances,
                              label=name, alpha=0.7)

        axes[1,0].set_title('Learning Stability (Variance)')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel(f'{window_size}-Episode Variance')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Cumulative reward
        for name, scores in results.items():
            cumsum = np.cumsum(scores)
            axes[1,1].plot(episodes, cumsum, linewidth=2, label=name)

        axes[1,1].set_title('Cumulative Reward')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Cumulative Reward')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_losses(self, policy_losses: List[float], value_losses: Optional[List[float]] = None,
                   title: str = "Training Losses"):
        """Plot training losses

        Args:
            policy_losses: Policy network losses
            value_losses: Value network losses (optional)
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(policy_losses, color='orange', alpha=0.7, label='Policy Loss')

        if value_losses is not None:
            ax.plot(value_losses, color='green', alpha=0.7, label='Value Loss')

        ax.set_title(title)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.show()

    def plot_network_analysis(self):
        """Create comprehensive network architecture analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        models = ['Tabular', 'Separate NN', 'Shared NN', 'Continuous NN']
        state_sizes = [10, 100, 1000, 10000]

        tabular_params = [s * 2 for s in state_sizes]  # Q-table size
        separate_params = [(s * 128 + 128 * 64 + 64 * 2) * 2 for s in state_sizes]
        shared_params = [s * 128 + 128 * 64 + 64 * 2 + 64 * 32 + 32 * 2 for s in state_sizes]
        continuous_params = [s * 128 + 128 * 64 + 64 * 4 for s in state_sizes]  # 2 actions

        x = np.arange(len(state_sizes))
        width = 0.2

        axes[0,0].bar(x - width*1.5, tabular_params, width, label='Tabular', alpha=0.8)
        axes[0,0].bar(x - width*0.5, separate_params, width, label='Separate NN', alpha=0.8)
        axes[0,0].bar(x + width*0.5, shared_params, width, label='Shared NN', alpha=0.8)
        axes[0,0].bar(x + width*1.5, continuous_params, width, label='Continuous NN', alpha=0.8)

        axes[0,0].set_title('Parameter Count vs State Size')
        axes[0,0].set_xlabel('State Size')
        axes[0,0].set_ylabel('Number of Parameters')
        axes[0,0].set_yscale('log')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([str(s) for s in state_sizes])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        episodes = np.arange(1000)

        tabular_curve = 100 * (1 - np.exp(-episodes / 200)) + np.random.normal(0, 5, 1000)
        separate_curve = 150 * (1 - np.exp(-episodes / 300)) + np.random.normal(0, 8, 1000)
        shared_curve = 180 * (1 - np.exp(-episodes / 250)) + np.random.normal(0, 6, 1000)

        axes[0,1].plot(episodes, tabular_curve, alpha=0.7, label='Tabular (small state)')
        axes[0,1].plot(episodes, separate_curve, alpha=0.7, label='Separate Networks')
        axes[0,1].plot(episodes, shared_curve, alpha=0.7, label='Shared Networks')

        axes[0,1].set_title('Learning Curves Comparison')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Average Return')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        sample_sizes = [1000, 5000, 10000, 50000]
        tabular_performance = [0.3, 0.8, 0.95, 0.98]
        nn_performance = [0.1, 0.4, 0.7, 0.9]
        shared_performance = [0.15, 0.5, 0.8, 0.95]

        axes[1,0].plot(sample_sizes, tabular_performance, 'o-',
                      label='Tabular', linewidth=2, markersize=8)
        axes[1,0].plot(sample_sizes, nn_performance, 's-',
                      label='Separate NN', linewidth=2, markersize=8)
        axes[1,0].plot(sample_sizes, shared_performance, '^-',
                      label='Shared NN', linewidth=2, markersize=8)

        axes[1,0].set_title('Sample Efficiency')
        axes[1,0].set_xlabel('Training Samples')
        axes[1,0].set_ylabel('Normalized Performance')
        axes[1,0].set_xscale('log')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        action_types = ['Discrete\\n(4 actions)', 'Discrete\\n(100 actions)',
                       'Continuous\\n(1D)', 'Continuous\\n(10D)']
        memory_requirements = [16, 400, 1, 10]  # Relative memory for action representation

        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        bars = axes[1,1].bar(action_types, memory_requirements, color=colors, alpha=0.8)

        axes[1,1].set_title('Action Space Memory Requirements')
        axes[1,1].set_ylabel('Relative Memory (log scale)')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)

        for bar, value in zip(bars, memory_requirements):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


def create_training_summary(results: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
    """Create training summary statistics

    Args:
        results: Training results dictionary
        algorithm_name: Name of the algorithm

    Returns:
        Summary statistics
    """
    scores = results.get('scores', [])

    if not scores:
        return {'error': 'No scores available'}

    summary = {
        'algorithm': algorithm_name,
        'total_episodes': len(scores),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'final_avg_score': np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores),
        'best_episode': np.max(scores),
        'convergence_episode': None
    }

    # Find convergence (when moving average stabilizes)
    if len(scores) >= 100:
        window = 20
        moving_avg = [np.mean(scores[i-window:i]) for i in range(window, len(scores))]
        threshold = 0.95 * np.max(moving_avg)

        for i, avg in enumerate(moving_avg):
            if avg >= threshold:
                summary['convergence_episode'] = i + window
                break

    return summary


def print_training_comparison(results_dict: Dict[str, Dict]) -> None:
    """Print comparison of training results

    Args:
        results_dict: Dictionary of algorithm results
    """
    print("Training Results Comparison:")
    print("=" * 50)

    summaries = []
    for alg_name, results in results_dict.items():
        summary = create_training_summary(results, alg_name)
        summaries.append(summary)

        print(f"\n{alg_name}:")
        print(f"  Final Average Score: {summary['final_avg_score']:.2f}")
        print(f"  Best Episode: {summary['best_episode']:.2f}")
        print(f"  Mean Score: {summary['mean_score']:.2f} ± {summary['std_score']:.2f}")

        if summary['convergence_episode']:
            print(f"  Converged at Episode: {summary['convergence_episode']}")

    # Overall comparison
    best_alg = max(summaries, key=lambda x: x['final_avg_score'])
    print(f"\nBest Performing Algorithm: {best_alg['algorithm']}")
    print(f"Final Score: {best_alg['final_avg_score']:.2f}")