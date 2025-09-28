"""
Utilities Module for CA19 Advanced RL Systems

This module provides utility functions, configuration management,
and helper classes for the advanced RL implementations in CA19.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import os
import time
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


@dataclass
class MissionConfig:
    """Configuration class for quantum RL missions"""

    # Quantum circuit parameters
    n_qubits: int = 6
    n_layers: int = 3
    quantum_shots: int = 1024

    # Agent parameters
    state_dim: int = 20
    action_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 32

    # Training parameters
    max_episodes: int = 500
    max_steps_per_episode: int = 1000
    epsilon_start: float = 0.3
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Mission parameters
    difficulty_level: str = "EXTREME"
    crisis_injection_rate: float = 0.2
    quantum_weight_adaptive: bool = True

    # Hardware parameters
    device: str = "auto"  # 'cpu', 'cuda', or 'auto'
    quantum_backend: str = "simulator"  # 'simulator' or 'hardware'

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class PerformanceTracker:
    """
    Comprehensive performance tracking for RL experiments

    Tracks various metrics across training episodes and provides
    analysis and visualization capabilities.
    """

    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.quantum_fidelity_history = []
        self.classical_loss_history = []
        self.quantum_loss_history = []
        self.td_errors = []
        self.dopamine_levels = []
        self.firing_rates = []
        self.entanglement_measures = []

        self.start_time = time.time()

    def update_episode(self, episode_reward: float, episode_length: int,
                      metrics: Dict[str, Any]):
        """Update tracking with episode results"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        # Extract quantum metrics
        if 'quantum_fidelity' in metrics:
            self.quantum_fidelity_history.append(metrics['quantum_fidelity'])
        if 'entanglement' in metrics:
            self.entanglement_measures.append(metrics['entanglement'])

        # Extract learning metrics
        if 'classical_loss' in metrics:
            self.classical_loss_history.append(metrics['classical_loss'])
        if 'quantum_loss' in metrics:
            self.quantum_loss_history.append(metrics['quantum_loss'])

        # Extract neuromorphic metrics
        if 'td_error' in metrics:
            self.td_errors.append(metrics['td_error'])
        if 'dopamine_level' in metrics:
            self.dopamine_levels.append(metrics['dopamine_level'])
        if 'avg_firing_rate' in metrics:
            self.firing_rates.append(metrics['avg_firing_rate'])

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of tracked metrics"""
        stats = {}

        if self.episode_rewards:
            stats.update({
                'total_episodes': len(self.episode_rewards),
                'avg_reward': np.mean(self.episode_rewards),
                'best_reward': np.max(self.episode_rewards),
                'worst_reward': np.min(self.episode_rewards),
                'reward_std': np.std(self.episode_rewards),
                'avg_episode_length': np.mean(self.episode_lengths)
            })

            # Learning progress (first 10% vs last 10%)
            if len(self.episode_rewards) >= 20:
                early_idx = len(self.episode_rewards) // 10
                late_idx = -early_idx if early_idx > 0 else None

                early_performance = np.mean(self.episode_rewards[:early_idx])
                late_performance = np.mean(self.episode_rewards[late_idx:])

                stats['learning_progress'] = (late_performance - early_performance) / (abs(early_performance) + 1e-6)

        # Quantum metrics
        if self.quantum_fidelity_history:
            stats.update({
                'avg_quantum_fidelity': np.mean(self.quantum_fidelity_history),
                'quantum_fidelity_std': np.std(self.quantum_fidelity_history),
                'final_quantum_fidelity': self.quantum_fidelity_history[-1]
            })

        if self.entanglement_measures:
            stats['avg_entanglement'] = np.mean(self.entanglement_measures)

        # Learning metrics
        if self.classical_loss_history:
            stats.update({
                'final_classical_loss': self.classical_loss_history[-1],
                'avg_classical_loss': np.mean(self.classical_loss_history[-50:])
            })

        if self.quantum_loss_history:
            stats['avg_quantum_loss'] = np.mean(self.quantum_loss_history[-50:])

        # Neuromorphic metrics
        if self.td_errors:
            stats.update({
                'avg_td_error': np.mean(self.td_errors),
                'final_td_error': self.td_errors[-1]
            })

        if self.dopamine_levels:
            stats.update({
                'avg_dopamine': np.mean(self.dopamine_levels),
                'final_dopamine': self.dopamine_levels[-1]
            })

        if self.firing_rates:
            stats.update({
                'avg_firing_rate': np.mean(self.firing_rates),
                'final_firing_rate': self.firing_rates[-1]
            })

        stats['training_time'] = time.time() - self.start_time

        return stats

    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot comprehensive training progress visualization"""
        if len(self.episode_rewards) < 5:
            print("Not enough data for plotting")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CA19 Advanced RL Training Progress', fontsize=16)

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.7, color='blue')
        if len(self.episode_rewards) >= 20:
            axes[0, 0].plot(np.convolve(self.episode_rewards, np.ones(20)/20, mode='same'),
                           color='darkblue', linewidth=2, label='Moving Average (20)')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Quantum fidelity
        if self.quantum_fidelity_history:
            axes[0, 1].plot(self.quantum_fidelity_history, alpha=0.7, color='purple')
            axes[0, 1].set_title('Quantum Fidelity')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Fidelity')
            axes[0, 1].grid(True)

        # Learning losses
        if self.classical_loss_history:
            axes[0, 2].plot(self.classical_loss_history, alpha=0.7, color='red', label='Classical')
            if self.quantum_loss_history:
                axes[0, 2].plot(self.quantum_loss_history, alpha=0.7, color='orange', label='Quantum')
            axes[0, 2].set_title('Learning Losses')
            axes[0, 2].set_xlabel('Update Step')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True)

        # Neuromorphic signals
        if self.td_errors:
            axes[1, 0].plot(self.td_errors, alpha=0.7, color='green', label='TD Error')
            axes[1, 0].set_title('Neuromorphic Learning Signals')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('TD Error')
            axes[1, 0].grid(True)

        if self.dopamine_levels:
            ax2 = axes[1, 0].twinx()
            ax2.plot(self.dopamine_levels, alpha=0.7, color='cyan', label='Dopamine')
            ax2.set_ylabel('Dopamine Level', color='cyan')
            ax2.tick_params(axis='y', labelcolor='cyan')

        # Firing rates and entanglement
        if self.firing_rates:
            axes[1, 1].plot(self.firing_rates, alpha=0.7, color='orange', label='Firing Rate')
            axes[1, 1].set_title('Neural Activity')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Avg Firing Rate (Hz)')
            axes[1, 1].grid(True)

        if self.entanglement_measures:
            ax2 = axes[1, 1].twinx()
            ax2.plot(self.entanglement_measures, alpha=0.7, color='purple', label='Entanglement')
            ax2.set_ylabel('Entanglement', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

        # Episode lengths
        axes[1, 2].plot(self.episode_lengths, alpha=0.7, color='brown')
        axes[1, 2].set_title('Episode Lengths')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps')
        axes[1, 2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")

        plt.show()

    def save_results(self, filepath: str):
        """Save tracking results to file"""
        results = {
            'summary_stats': self.get_summary_stats(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'quantum_fidelity_history': self.quantum_fidelity_history,
            'classical_loss_history': self.classical_loss_history,
            'quantum_loss_history': self.quantum_loss_history,
            'td_errors': self.td_errors,
            'dopamine_levels': self.dopamine_levels,
            'firing_rates': self.firing_rates,
            'entanglement_measures': self.entanglement_measures,
            'training_time': time.time() - self.start_time
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str):
        """Load tracking results from file"""
        with open(filepath, 'r') as f:
            results = json.load(f)

        self.episode_rewards = results.get('episode_rewards', [])
        self.episode_lengths = results.get('episode_lengths', [])
        self.quantum_fidelity_history = results.get('quantum_fidelity_history', [])
        self.classical_loss_history = results.get('classical_loss_history', [])
        self.quantum_loss_history = results.get('quantum_loss_history', [])
        self.td_errors = results.get('td_errors', [])
        self.dopamine_levels = results.get('dopamine_levels', [])
        self.firing_rates = results.get('firing_rates', [])
        self.entanglement_measures = results.get('entanglement_measures', [])

        print(f"Results loaded from {filepath}")


class ExperimentManager:
    """
    Manager for running and comparing multiple RL experiments

    Provides systematic evaluation of different algorithms and configurations.
    """

    def __init__(self, base_config: MissionConfig):
        self.base_config = base_config
        self.experiments = []
        self.results = {}

    def add_experiment(self, name: str, config_overrides: Dict[str, Any],
                      agent_class: Any, environment_class: Any):
        """Add an experiment to the test suite"""
        experiment = {
            'name': name,
            'config': {**asdict(self.base_config), **config_overrides},
            'agent_class': agent_class,
            'environment_class': environment_class,
            'tracker': PerformanceTracker()
        }
        self.experiments.append(experiment)

    def run_experiment(self, experiment_idx: int, num_episodes: int = 50,
                      verbose: bool = False) -> Dict[str, Any]:
        """Run a specific experiment"""
        if experiment_idx >= len(self.experiments):
            raise ValueError(f"Experiment {experiment_idx} not found")

        experiment = self.experiments[experiment_idx]
        config = experiment['config']
        tracker = experiment['tracker']

        print(f"🚀 Running Experiment: {experiment['name']}")
        print(f"Configuration: {config}")

        # Initialize agent and environment
        agent = experiment['agent_class'](
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            learning_rate=config['learning_rate']
        )

        environment = experiment['environment_class'](
            difficulty_level=config['difficulty_level']
        )

        # Training loop
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < config['max_steps_per_episode']:
                epsilon = max(config['epsilon_min'],
                            config['epsilon_start'] * (config['epsilon_decay'] ** episode))

                action, action_info = agent.select_action(state, epsilon=epsilon)
                next_state, reward, done, info = environment.step(action)

                agent.store_experience(state, action, reward, next_state, done)
                learning_metrics = agent.learn()

                episode_reward += reward
                episode_length += 1
                state = next_state

            # Update tracking
            combined_metrics = {**action_info, **learning_metrics, **info}
            tracker.update_episode(episode_reward, episode_length, combined_metrics)

            if verbose and episode % 10 == 0:
                print(f"Episode {episode}: Reward={episode_reward:.2f}, Length={episode_length}")

        results = {
            'experiment_name': experiment['name'],
            'config': config,
            'summary_stats': tracker.get_summary_stats(),
            'full_tracker': tracker
        }

        self.results[experiment['name']] = results
        return results

    def run_all_experiments(self, num_episodes: int = 50) -> Dict[str, Any]:
        """Run all experiments in the suite"""
        all_results = {}

        for i, experiment in enumerate(self.experiments):
            print(f"\n{'='*60}")
            result = self.run_experiment(i, num_episodes=num_episodes, verbose=True)
            all_results[experiment['name']] = result

        return all_results

    def compare_experiments(self, metric: str = 'avg_reward') -> Dict[str, Any]:
        """Compare results across experiments"""
        if not self.results:
            return {'error': 'No experiment results available'}

        comparison = {}
        for exp_name, result in self.results.items():
            comparison[exp_name] = result['summary_stats'].get(metric, 0)

        # Sort by performance
        sorted_experiments = sorted(comparison.items(), key=lambda x: x[1], reverse=True)

        return {
            'ranking': sorted_experiments,
            'best_experiment': sorted_experiments[0][0],
            'best_score': sorted_experiments[0][1],
            'metric': metric
        }

    def generate_comparison_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive comparison report"""
        if not self.results:
            return "No experiments to compare"

        report = []
        report.append("🎯 CA19 Advanced RL Experiment Comparison Report")
        report.append("=" * 60)

        # Summary statistics
        comparison = self.compare_experiments()
        report.append(f"Best Performing Experiment: {comparison['best_experiment']}")
        report.append(f"Best {comparison['metric']}: {comparison['best_score']:.4f}")
        report.append("")

        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 30)

        for exp_name, result in self.results.items():
            stats = result['summary_stats']
            report.append(f"\n{exp_name}:")
            report.append(f"  Episodes: {stats.get('total_episodes', 0)}")
            report.append(f"  Avg Reward: {stats.get('avg_reward', 0):.2f}")
            report.append(f"  Best Reward: {stats.get('best_reward', 0):.2f}")
            report.append(f"  Training Time: {stats.get('training_time', 0):.2f}s")

            if 'avg_quantum_fidelity' in stats:
                report.append(f"  Quantum Fidelity: {stats['avg_quantum_fidelity']:.4f}")
            if 'avg_td_error' in stats:
                report.append(f"  Avg TD Error: {stats['avg_td_error']:.4f}")

        report.append("\n" + "=" * 60)

        final_report = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(final_report)
            print(f"Comparison report saved to {save_path}")

        return final_report


def create_default_config() -> MissionConfig:
    """Create default mission configuration"""
    return MissionConfig()


def save_config(config: MissionConfig, filepath: str):
    """Save configuration to file"""
    with open(filepath, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> MissionConfig:
    """Load configuration from file"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return MissionConfig(**config_dict)


def setup_experiment_logging(log_dir: str = "experiments"):
    """Setup logging directory for experiments"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created experiment logging directory: {log_dir}")
    return log_dir


def benchmark_quantum_vs_classical(quantum_agent, classical_agent,
                                  environment, num_episodes: int = 50) -> Dict[str, Any]:
    """
    Benchmark quantum-enhanced vs classical RL performance

    Returns comprehensive comparison metrics
    """
    print("🏁 Starting Quantum vs Classical Benchmark")

    # Test quantum agent
    print("⚛️  Testing Quantum-Enhanced Agent...")
    quantum_rewards = []
    quantum_metrics = []

    for episode in range(num_episodes):
        state = environment.reset()
        episode_reward = 0
        done = False

        while not done:
            action, action_info = quantum_agent.select_action(state, epsilon=0.1, quantum_enabled=True)
            next_state, reward, done, info = environment.step(action)
            episode_reward += reward
            state = next_state

        quantum_rewards.append(episode_reward)
        quantum_metrics.append(action_info)

    # Test classical agent
    print("🧠 Testing Classical Agent...")
    classical_rewards = []
    classical_metrics = []

    for episode in range(num_episodes):
        state = environment.reset()
        episode_reward = 0
        done = False

        while not done:
            action, action_info = classical_agent.select_action(state, epsilon=0.1, quantum_enabled=False)
            next_state, reward, done, info = environment.step(action)
            episode_reward += reward
            state = next_state

        classical_rewards.append(episode_reward)
        classical_metrics.append(action_info)

    # Statistical analysis
    quantum_avg = np.mean(quantum_rewards)
    classical_avg = np.mean(classical_rewards)
    quantum_std = np.std(quantum_rewards)
    classical_std = np.std(classical_rewards)

    # Statistical significance test
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(quantum_rewards, classical_rewards)
        significant = p_value < 0.05
    except ImportError:
        # Simple difference test
        diff = abs(quantum_avg - classical_avg)
        pooled_std = np.sqrt((quantum_std**2 + classical_std**2) / 2)
        t_stat = diff / (pooled_std / np.sqrt(len(quantum_rewards)))
        p_value = 0.05  # Conservative estimate
        significant = diff > 2 * pooled_std

    advantage_percent = ((quantum_avg - classical_avg) / abs(classical_avg)) * 100

    results = {
        'quantum_avg_reward': quantum_avg,
        'classical_avg_reward': classical_avg,
        'quantum_std': quantum_std,
        'classical_std': classical_std,
        'advantage_percent': advantage_percent,
        'statistical_significance': significant,
        'p_value': p_value,
        't_statistic': t_stat,
        'quantum_superior': quantum_avg > classical_avg,
        'all_quantum_rewards': quantum_rewards,
        'all_classical_rewards': classical_rewards
    }

    print("✅ Benchmark Complete!")
    print(f"Quantum Avg: {quantum_avg:.2f} ± {quantum_std:.2f}")
    print(f"Classical Avg: {classical_avg:.2f} ± {classical_std:.2f}")
    print(f"Advantage: {advantage_percent:.1f}%")
    print(f"Statistically Significant: {'YES' if significant else 'NO'}")

    return results