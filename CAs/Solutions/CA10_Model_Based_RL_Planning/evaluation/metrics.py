"""
Performance metrics for model-based RL evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import time


class PerformanceMetrics:
    """Comprehensive performance metrics for model-based RL methods"""

    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}

    def calculate_episode_metrics(self, episode_rewards: List[float], 
                                episode_lengths: List[int]) -> Dict[str, float]:
        """Calculate metrics for a single episode"""
        return {
            'reward': episode_rewards[-1] if episode_rewards else 0.0,
            'length': episode_lengths[-1] if episode_lengths else 0,
            'cumulative_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
        }

    def calculate_learning_metrics(self, episode_rewards: List[float]) -> Dict[str, float]:
        """Calculate learning performance metrics"""
        if not episode_rewards:
            return {}

        # Basic statistics
        final_performance = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        initial_performance = np.mean(episode_rewards[:10]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        
        # Learning efficiency (area under learning curve)
        learning_efficiency = np.sum(episode_rewards) / len(episode_rewards)
        
        # Sample efficiency (episodes to reach target performance)
        target_performance = 0.8 * final_performance
        sample_efficiency = self._episodes_to_target(episode_rewards, target_performance)
        
        # Stability (inverse of variance in final episodes)
        final_episodes = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
        stability = 1.0 / (np.var(final_episodes) + 1e-8)
        
        # Convergence speed (episodes to reach 95% of final performance)
        convergence_episodes = self._episodes_to_target(episode_rewards, 0.95 * final_performance)
        
        # Improvement ratio
        improvement_ratio = (final_performance - initial_performance) / (abs(initial_performance) + 1e-8)
        
        return {
            'final_performance': final_performance,
            'initial_performance': initial_performance,
            'learning_efficiency': learning_efficiency,
            'sample_efficiency': sample_efficiency,
            'stability': stability,
            'convergence_episodes': convergence_episodes,
            'improvement_ratio': improvement_ratio,
            'total_episodes': len(episode_rewards),
        }

    def calculate_model_metrics(self, model_accuracy: float, 
                              model_uncertainty: float = 0.0) -> Dict[str, float]:
        """Calculate model-specific metrics"""
        return {
            'model_accuracy': model_accuracy,
            'model_uncertainty': model_uncertainty,
            'model_confidence': 1.0 - model_uncertainty,
        }

    def calculate_planning_metrics(self, planning_times: List[float],
                                 planning_costs: List[float]) -> Dict[str, float]:
        """Calculate planning-specific metrics"""
        if not planning_times:
            return {}

        return {
            'avg_planning_time': np.mean(planning_times),
            'std_planning_time': np.std(planning_times),
            'max_planning_time': np.max(planning_times),
            'min_planning_time': np.min(planning_times),
            'avg_planning_cost': np.mean(planning_costs) if planning_costs else 0.0,
            'total_planning_time': np.sum(planning_times),
            'planning_efficiency': 1.0 / (np.mean(planning_times) + 1e-8),
        }

    def calculate_computational_metrics(self, training_times: List[float],
                                      memory_usage: List[float] = None) -> Dict[str, float]:
        """Calculate computational efficiency metrics"""
        metrics = {
            'avg_training_time': np.mean(training_times),
            'total_training_time': np.sum(training_times),
            'std_training_time': np.std(training_times),
        }
        
        if memory_usage:
            metrics.update({
                'avg_memory_usage': np.mean(memory_usage),
                'max_memory_usage': np.max(memory_usage),
                'memory_efficiency': 1.0 / (np.mean(memory_usage) + 1e-8),
            })
            
        return metrics

    def _episodes_to_target(self, rewards: List[float], target: float) -> int:
        """Calculate episodes needed to reach target performance"""
        for i in range(10, len(rewards)):
            if np.mean(rewards[max(0, i-9):i+1]) >= target:
                return i + 1
        return len(rewards)

    def aggregate_metrics(self, multiple_runs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across multiple runs"""
        if not multiple_runs:
            return {}

        # Get all metric names
        all_keys = set()
        for run in multiple_runs:
            all_keys.update(run.keys())

        aggregated = {}
        for key in all_keys:
            values = [run.get(key, 0) for run in multiple_runs]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
                aggregated[f'min_{key}'] = np.min(values)
                aggregated[f'max_{key}'] = np.max(values)

        return aggregated

    def create_metrics_report(self, metrics_dict: Dict[str, Any]) -> str:
        """Create a formatted metrics report"""
        report = "📊 Performance Metrics Report\n"
        report += "=" * 40 + "\n\n"

        # Learning metrics
        if any(key.startswith(('final_performance', 'learning_efficiency', 'sample_efficiency')) 
               for key in metrics_dict.keys()):
            report += "🎯 Learning Performance:\n"
            report += f"  Final Performance: {metrics_dict.get('final_performance', 0):.3f}\n"
            report += f"  Learning Efficiency: {metrics_dict.get('learning_efficiency', 0):.3f}\n"
            report += f"  Sample Efficiency: {metrics_dict.get('sample_efficiency', 0):.1f} episodes\n"
            report += f"  Stability: {metrics_dict.get('stability', 0):.3f}\n"
            report += f"  Improvement Ratio: {metrics_dict.get('improvement_ratio', 0):.2f}\n\n"

        # Planning metrics
        if any(key.startswith('planning') for key in metrics_dict.keys()):
            report += "🧠 Planning Performance:\n"
            report += f"  Avg Planning Time: {metrics_dict.get('avg_planning_time', 0):.4f}s\n"
            report += f"  Planning Efficiency: {metrics_dict.get('planning_efficiency', 0):.2f}\n"
            report += f"  Total Planning Time: {metrics_dict.get('total_planning_time', 0):.2f}s\n\n"

        # Computational metrics
        if any(key.startswith(('training_time', 'memory')) for key in metrics_dict.keys()):
            report += "💻 Computational Performance:\n"
            report += f"  Avg Training Time: {metrics_dict.get('avg_training_time', 0):.4f}s\n"
            report += f"  Total Training Time: {metrics_dict.get('total_training_time', 0):.2f}s\n"
            if 'avg_memory_usage' in metrics_dict:
                report += f"  Avg Memory Usage: {metrics_dict.get('avg_memory_usage', 0):.2f}MB\n\n"

        # Model metrics
        if any(key.startswith('model') for key in metrics_dict.keys()):
            report += "🎲 Model Performance:\n"
            report += f"  Model Accuracy: {metrics_dict.get('model_accuracy', 0):.3f}\n"
            report += f"  Model Confidence: {metrics_dict.get('model_confidence', 0):.3f}\n\n"

        return report

    def visualize_metrics(self, metrics_history: List[Dict[str, float]], 
                         save_path: str = None) -> plt.Figure:
        """Create visualization of metrics over time"""
        if not metrics_history:
            return None

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(metrics_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics Over Time', fontsize=16)

        # Learning performance
        if 'final_performance' in df.columns:
            axes[0, 0].plot(df.index, df['final_performance'], 'b-', linewidth=2, label='Final Performance')
            axes[0, 0].set_title('Learning Performance')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Performance')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # Learning efficiency
        if 'learning_efficiency' in df.columns:
            axes[0, 1].plot(df.index, df['learning_efficiency'], 'g-', linewidth=2, label='Learning Efficiency')
            axes[0, 1].set_title('Learning Efficiency')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Efficiency')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Planning metrics
        if 'avg_planning_time' in df.columns:
            axes[1, 0].plot(df.index, df['avg_planning_time'], 'r-', linewidth=2, label='Planning Time')
            axes[1, 0].set_title('Planning Performance')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Stability
        if 'stability' in df.columns:
            axes[1, 1].plot(df.index, df['stability'], 'm-', linewidth=2, label='Stability')
            axes[1, 1].set_title('Learning Stability')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Stability')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def demonstrate_metrics():
    """Demonstrate the metrics calculation"""
    print("Performance Metrics Demonstration")
    print("=" * 40)

    # Create sample data
    episode_rewards = np.random.normal(0.5, 0.2, 100).cumsum() / 10
    episode_lengths = np.random.randint(10, 50, 100)
    planning_times = np.random.exponential(0.1, 100)

    # Create metrics calculator
    metrics_calc = PerformanceMetrics()

    # Calculate learning metrics
    learning_metrics = metrics_calc.calculate_learning_metrics(episode_rewards)
    print("Learning Metrics:")
    for key, value in learning_metrics.items():
        print(f"  {key}: {value:.3f}")

    # Calculate planning metrics
    planning_metrics = metrics_calc.calculate_planning_metrics(planning_times)
    print("\nPlanning Metrics:")
    for key, value in planning_metrics.items():
        print(f"  {key}: {value:.3f}")

    # Create report
    all_metrics = {**learning_metrics, **planning_metrics}
    report = metrics_calc.create_metrics_report(all_metrics)
    print(f"\n{report}")

    print("✅ Metrics demonstration complete!")


if __name__ == "__main__":
    demonstrate_metrics()
