"""
Experiment runner for CA1 Deep RL Fundamentals.

This module contains functions for running comprehensive experiments
and generating detailed reports and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import time

from ..agents.ca1_agents import DQNAgent, REINFORCEAgent, ActorCriticAgent
from ..environments.custom_envs import create_cartpole_env
from ..evaluation.evaluators import AgentEvaluator, compare_agents, plot_comparison
from ..utils.ca1_utils import set_seed, moving_average


class ExperimentRunner:
    """
    A comprehensive experiment runner for comparing different RL algorithms.
    """
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.plots_dir = os.path.join(base_dir, "plots")
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.experiment_results = {}
        
    def run_comprehensive_comparison(self, 
                                   n_training_episodes: int = 1000,
                                   n_eval_episodes: int = 100,
                                   n_runs: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all algorithms.
        
        Args:
            n_training_episodes: Number of training episodes per run
            n_eval_episodes: Number of evaluation episodes
            n_runs: Number of independent runs for statistical significance
            
        Returns:
            Dictionary containing all experiment results
        """
        print("=" * 60)
        print("COMPREHENSIVE RL ALGORITHM COMPARISON")
        print("=" * 60)
        
        # Set up environments and agents
        env = create_cartpole_env()
        
        agent_configs = {
            'DQN': {
                'state_size': 4,
                'action_size': 2,
                'use_dueling': True,
                'use_double_dqn': True,
                'lr': 1e-3
            },
            'REINFORCE': {
                'state_size': 4,
                'action_size': 2,
                'lr': 1e-3
            },
            'Actor-Critic': {
                'state_size': 4,
                'action_size': 2,
                'lr_actor': 1e-3,
                'lr_critic': 1e-3
            }
        }
        
        # Run experiments for each algorithm
        all_results = {}
        
        for algorithm_name, config in agent_configs.items():
            print(f"\nRunning {algorithm_name} experiments...")
            algorithm_results = self._run_algorithm_experiments(
                algorithm_name, config, env, n_training_episodes, n_eval_episodes, n_runs
            )
            all_results[algorithm_name] = algorithm_results
            
        self.experiment_results = all_results
        
        # Generate comprehensive analysis
        self._generate_comprehensive_analysis()
        
        return all_results
    
    def _run_algorithm_experiments(self, 
                                 algorithm_name: str,
                                 config: Dict[str, Any],
                                 env,
                                 n_training_episodes: int,
                                 n_eval_episodes: int,
                                 n_runs: int) -> Dict[str, Any]:
        """Run experiments for a single algorithm across multiple runs."""
        
        training_scores = []
        evaluation_results = []
        training_times = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}")
            
            # Set seed for reproducibility
            set_seed(42 + run)
            
            # Create fresh environment and agent
            env_run = create_cartpole_env()
            
            if algorithm_name == 'DQN':
                agent = DQNAgent(**config)
                start_time = time.time()
                scores = self._train_dqn_agent(agent, env_run, n_training_episodes)
            elif algorithm_name == 'REINFORCE':
                agent = REINFORCEAgent(**config)
                start_time = time.time()
                scores = self._train_reinforce_agent(agent, env_run, n_training_episodes)
            elif algorithm_name == 'Actor-Critic':
                agent = ActorCriticAgent(**config)
                start_time = time.time()
                scores = self._train_actor_critic_agent(agent, env_run, n_training_episodes)
            
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Evaluate the trained agent
            evaluator = AgentEvaluator(agent, env_run, n_eval_episodes)
            eval_results = evaluator.evaluate()
            
            training_scores.append(scores)
            evaluation_results.append(eval_results)
            
            env_run.close()
        
        # Aggregate results
        return {
            'training_scores': training_scores,
            'evaluation_results': evaluation_results,
            'training_times': training_times,
            'mean_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times)
        }
    
    def _train_dqn_agent(self, agent: DQNAgent, env, n_episodes: int) -> List[float]:
        """Train DQN agent and return scores."""
        from ..agents.ca1_agents import train_dqn_agent
        return train_dqn_agent(agent, env, n_episodes=n_episodes, max_t=200)
    
    def _train_reinforce_agent(self, agent: REINFORCEAgent, env, n_episodes: int) -> List[float]:
        """Train REINFORCE agent and return scores."""
        from ..agents.ca1_agents import train_reinforce_agent
        return train_reinforce_agent(agent, env, n_episodes=n_episodes, max_t=200)
    
    def _train_actor_critic_agent(self, agent: ActorCriticAgent, env, n_episodes: int) -> List[float]:
        """Train Actor-Critic agent and return scores."""
        from ..agents.ca1_agents import train_actor_critic_agent
        return train_actor_critic_agent(agent, env, n_episodes=n_episodes, max_t=200)
    
    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis and visualizations."""
        print("\nGenerating comprehensive analysis...")
        
        # 1. Training curves comparison
        self._plot_training_curves()
        
        # 2. Final performance comparison
        self._plot_final_performance()
        
        # 3. Sample efficiency analysis
        self._plot_sample_efficiency()
        
        # 4. Statistical significance tests
        self._perform_statistical_tests()
        
        # 5. Save detailed results
        self._save_detailed_results()
        
        print(f"Analysis complete! Results saved to {self.base_dir}")
    
    def _plot_training_curves(self):
        """Plot training curves for all algorithms."""
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green']
        algorithms = list(self.experiment_results.keys())
        
        for i, (algorithm, results) in enumerate(self.experiment_results.items()):
            # Average training scores across runs
            all_scores = results['training_scores']
            min_length = min(len(scores) for scores in all_scores)
            averaged_scores = np.mean([scores[:min_length] for scores in all_scores], axis=0)
            std_scores = np.std([scores[:min_length] for scores in all_scores], axis=0)
            
            # Plot mean with confidence interval
            episodes = range(len(averaged_scores))
            plt.plot(episodes, averaged_scores, color=colors[i], label=f'{algorithm} (mean)', linewidth=2)
            plt.fill_between(episodes, 
                           averaged_scores - std_scores, 
                           averaged_scores + std_scores, 
                           alpha=0.3, color=colors[i])
            
            # Plot moving average
            if len(averaged_scores) >= 50:
                ma = moving_average(averaged_scores, window=50)
                plt.plot(range(49, len(averaged_scores)), ma, '--', 
                        color=colors[i], alpha=0.8, linewidth=1)
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Training Curves Comparison (Mean ± Std across runs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=195, color='black', linestyle=':', alpha=0.7, label='CartPole Threshold')
        
        save_path = os.path.join(self.plots_dir, 'training_curves_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves plot saved to {save_path}")
    
    def _plot_final_performance(self):
        """Plot final performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        algorithms = list(self.experiment_results.keys())
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        # Extract final evaluation metrics
        mean_scores = []
        std_scores = []
        success_rates = []
        training_times = []
        
        for algorithm, results in self.experiment_results.items():
            eval_results = results['evaluation_results']
            scores = [run['mean_score'] for run in eval_results]
            success_rates_run = [run['success_rate'] for run in eval_results]
            
            mean_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
            success_rates.append(np.mean(success_rates_run))
            training_times.append(results['mean_training_time'])
        
        # Mean scores comparison
        bars = axes[0, 0].bar(algorithms, mean_scores, yerr=std_scores, 
                             capsize=5, alpha=0.7, color=colors)
        axes[0, 0].set_ylabel('Mean Score')
        axes[0, 0].set_title('Final Performance Comparison')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, score in zip(bars, mean_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Success rates comparison
        bars = axes[0, 1].bar(algorithms, success_rates, alpha=0.7, color=colors)
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_title('Success Rate Comparison')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, success_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Training time comparison
        bars = axes[1, 0].bar(algorithms, training_times, alpha=0.7, color=colors)
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, time_val in zip(bars, training_times):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Performance summary table
        summary_data = []
        for i, algorithm in enumerate(algorithms):
            summary_data.append([
                algorithm,
                f"{mean_scores[i]:.2f} ± {std_scores[i]:.2f}",
                f"{success_rates[i]:.1%}",
                f"{training_times[i]:.1f}s"
            ])
        
        table = axes[1, 1].table(cellText=summary_data,
                               colLabels=['Algorithm', 'Mean Score ± Std', 'Success Rate', 'Training Time'],
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, 'final_performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Final performance plot saved to {save_path}")
    
    def _plot_sample_efficiency(self):
        """Plot sample efficiency analysis."""
        plt.figure(figsize=(12, 8))
        
        algorithms = list(self.experiment_results.keys())
        colors = ['blue', 'red', 'green']
        
        for i, (algorithm, results) in enumerate(self.experiment_results.items()):
            # Calculate episodes to threshold for each run
            episodes_to_threshold = []
            
            for run_scores in results['training_scores']:
                if len(run_scores) >= 100:
                    # Calculate moving average
                    ma = moving_average(run_scores, window=100)
                    
                    # Find first episode where threshold is reached
                    threshold_episode = None
                    for j, avg_score in enumerate(ma):
                        if avg_score >= 195.0:
                            threshold_episode = j + 100
                            break
                    
                    episodes_to_threshold.append(threshold_episode)
            
            if episodes_to_threshold:
                episodes_to_threshold = [ep for ep in episodes_to_threshold if ep is not None]
                if episodes_to_threshold:
                    plt.hist(episodes_to_threshold, bins=20, alpha=0.6, 
                            label=f'{algorithm} (mean: {np.mean(episodes_to_threshold):.0f})',
                            color=colors[i])
        
        plt.xlabel('Episodes to Reach Threshold (195)')
        plt.ylabel('Frequency')
        plt.title('Sample Efficiency: Episodes to Reach CartPole Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, 'sample_efficiency_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sample efficiency plot saved to {save_path}")
    
    def _perform_statistical_tests(self):
        """Perform statistical significance tests."""
        print("\nPerforming statistical significance tests...")
        
        # Extract final scores for each algorithm
        algorithm_scores = {}
        for algorithm, results in self.experiment_results.items():
            eval_results = results['evaluation_results']
            scores = [run['mean_score'] for run in eval_results]
            algorithm_scores[algorithm] = scores
        
        # Perform pairwise t-tests
        from scipy import stats
        
        algorithms = list(algorithm_scores.keys())
        n_algorithms = len(algorithms)
        
        print("\nPairwise t-test results (final scores):")
        print("-" * 50)
        
        for i in range(n_algorithms):
            for j in range(i + 1, n_algorithms):
                alg1, alg2 = algorithms[i], algorithms[j]
                scores1, scores2 = algorithm_scores[alg1], algorithm_scores[alg2]
                
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"{alg1} vs {alg2}: t={t_stat:.3f}, p={p_value:.4f} {significance}")
    
    def _save_detailed_results(self):
        """Save detailed results to JSON files."""
        # Save raw results
        raw_results_path = os.path.join(self.results_dir, 'raw_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for algorithm, results in self.experiment_results.items():
            serializable_results[algorithm] = {}
            for key, value in results.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_results[algorithm][key] = [arr.tolist() for arr in value]
                elif isinstance(value, np.ndarray):
                    serializable_results[algorithm][key] = value.tolist()
                else:
                    serializable_results[algorithm][key] = value
        
        with open(raw_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary statistics
        summary_stats = {}
        for algorithm, results in self.experiment_results.items():
            eval_results = results['evaluation_results']
            
            summary_stats[algorithm] = {
                'mean_score': float(np.mean([run['mean_score'] for run in eval_results])),
                'std_score': float(np.std([run['mean_score'] for run in eval_results])),
                'mean_success_rate': float(np.mean([run['success_rate'] for run in eval_results])),
                'mean_training_time': float(results['mean_training_time']),
                'std_training_time': float(results['std_training_time'])
            }
        
        summary_path = os.path.join(self.results_dir, 'summary_statistics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Detailed results saved to {self.results_dir}")


def run_quick_demo():
    """Run a quick demonstration of the experiment runner."""
    print("Running quick demo experiment...")
    
    runner = ExperimentRunner(base_dir="demo_experiments")
    results = runner.run_comprehensive_comparison(
        n_training_episodes=100,
        n_eval_episodes=20,
        n_runs=2
    )
    
    print("Demo experiment completed!")
    return results


if __name__ == "__main__":
    # Run the experiment
    runner = ExperimentRunner()
    results = runner.run_comprehensive_comparison(
        n_training_episodes=500,
        n_eval_episodes=50,
        n_runs=3
    )
