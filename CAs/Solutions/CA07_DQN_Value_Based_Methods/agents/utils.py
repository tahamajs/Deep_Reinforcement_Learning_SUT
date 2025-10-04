"""
DQN Utilities and Visualization Tools for CA07
===============================================
This module provides utility functions for visualization, analysis, and debugging DQN agents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from collections import defaultdict
import json
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DQNVisualizer:
    """Visualization tools for DQN analysis"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_learning_curves(self, results: Dict[str, Dict], 
                           window: int = 20, save_name: str = "learning_curves.png"):
        """Plot learning curves for multiple agents"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        # Learning curves
        ax1 = axes[0, 0]
        for i, (agent_name, result) in enumerate(results.items()):
            scores = result['scores']
            smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, label=agent_name, color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_title('Learning Curves Comparison')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Smoothed Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss curves
        ax2 = axes[0, 1]
        for i, (agent_name, result) in enumerate(results.items()):
            if 'losses' in result:
                losses = result['losses']
                smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax2.plot(smoothed_losses, label=agent_name, color=colors[i % len(colors)], linewidth=2)
        
        ax2.set_title('Loss Curves Comparison')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Smoothed Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final performance comparison
        ax3 = axes[1, 0]
        agent_names = list(results.keys())
        final_scores = [np.mean(result['scores'][-50:]) for result in results.values()]
        bars = ax3.bar(agent_names, final_scores, alpha=0.7, color=colors[:len(agent_names)])
        ax3.set_title('Final Performance Comparison')
        ax3.set_ylabel('Final Average Score (Last 50 episodes)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, final_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Training stability
        ax4 = axes[1, 1]
        stability_scores = [np.std(result['scores'][-100:]) for result in results.values()]
        bars = ax4.bar(agent_names, stability_scores, alpha=0.7, color='orange')
        ax4.set_title('Training Stability (Lower is Better)')
        ax4.set_ylabel('Score Standard Deviation')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, stability_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_q_value_analysis(self, agent, env, num_states: int = 100, 
                            save_name: str = "q_value_analysis.png"):
        """Analyze Q-value distributions and patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect Q-values from random states
        q_values_list = []
        states_list = []
        
        state, _ = env.reset()
        for _ in range(num_states):
            q_values = agent.get_q_values(state)
            q_values_list.append(q_values)
            states_list.append(state.copy())
            
            action = agent.select_action(state, epsilon=0.0)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        
        q_values_array = np.array(q_values_list)
        
        # Q-value distribution
        ax1 = axes[0, 0]
        for action in range(q_values_array.shape[1]):
            ax1.hist(q_values_array[:, action], alpha=0.6, 
                    label=f'Action {action}', bins=20)
        ax1.set_title('Q-value Distribution by Action')
        ax1.set_xlabel('Q-value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-value correlation matrix
        ax2 = axes[0, 1]
        correlation_matrix = np.corrcoef(q_values_array.T)
        im = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('Q-value Correlation Matrix')
        ax2.set_xlabel('Action')
        ax2.set_ylabel('Action')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2)
        
        # Add correlation values
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                        ha='center', va='center', color='white')
        
        # Q-value range over time
        ax3 = axes[1, 0]
        q_max = np.max(q_values_array, axis=1)
        q_min = np.min(q_values_array, axis=1)
        q_mean = np.mean(q_values_array, axis=1)
        
        ax3.plot(q_max, label='Max Q-value', alpha=0.7)
        ax3.plot(q_min, label='Min Q-value', alpha=0.7)
        ax3.plot(q_mean, label='Mean Q-value', alpha=0.7)
        ax3.fill_between(range(len(q_max)), q_min, q_max, alpha=0.2)
        ax3.set_title('Q-value Range Over States')
        ax3.set_xlabel('State Index')
        ax3.set_ylabel('Q-value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Action selection frequency
        ax4 = axes[1, 1]
        action_counts = defaultdict(int)
        for q_values in q_values_array:
            best_action = np.argmax(q_values)
            action_counts[best_action] += 1
        
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        ax4.bar(actions, counts, alpha=0.7, color='green')
        ax4.set_title('Action Selection Frequency')
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_hyperparameter_sensitivity(self, results: Dict[str, float], 
                                      param_name: str, save_name: str = "hyperparameter_sensitivity.png"):
        """Plot hyperparameter sensitivity analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        param_values = list(results.keys())
        scores = list(results.values())
        
        # Bar plot
        ax1 = axes[0]
        bars = ax1.bar(range(len(param_values)), scores, alpha=0.7, color='blue')
        ax1.set_title(f'{param_name} Sensitivity Analysis')
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Final Average Score')
        ax1.set_xticks(range(len(param_values)))
        ax1.set_xticklabels(param_values, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Line plot (if numeric)
        ax2 = axes[1]
        try:
            numeric_values = [float(v) for v in param_values]
            ax2.plot(numeric_values, scores, 'o-', linewidth=2, markersize=8, color='red')
            ax2.set_title(f'{param_name} Sensitivity (Line Plot)')
            ax2.set_xlabel(param_name)
            ax2.set_ylabel('Final Average Score')
            ax2.grid(True, alpha=0.3)
        except ValueError:
            # If not numeric, use bar plot
            ax2.bar(range(len(param_values)), scores, alpha=0.7, color='red')
            ax2.set_title(f'{param_name} Sensitivity (Alternative)')
            ax2.set_xlabel(param_name)
            ax2.set_ylabel('Final Average Score')
            ax2.set_xticks(range(len(param_values)))
            ax2.set_xticklabels(param_values, rotation=45)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_robustness_analysis(self, robustness_results: Dict[str, Dict], 
                                save_name: str = "robustness_analysis.png"):
        """Plot robustness analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Seed robustness
        if 'seed_robustness' in robustness_results:
            ax1 = axes[0, 0]
            seed_data = robustness_results['seed_robustness']
            seed_names = list(seed_data.keys())
            seed_scores = list(seed_data.values())
            
            bars = ax1.bar(seed_names, seed_scores, alpha=0.7, color='blue')
            ax1.set_title('Robustness to Random Seeds')
            ax1.set_xlabel('Seed')
            ax1.set_ylabel('Final Average Score')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            mean_score = np.mean(seed_scores)
            std_score = np.std(seed_scores)
            ax1.axhline(mean_score, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_score:.1f}')
            ax1.axhline(mean_score + std_score, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_score + std_score:.1f}')
            ax1.axhline(mean_score - std_score, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_score - std_score:.1f}')
            ax1.legend()
        
        # Reward scale robustness
        if 'scale_robustness' in robustness_results:
            ax2 = axes[0, 1]
            scale_data = robustness_results['scale_robustness']
            scales = [float(k.split('_')[1]) for k in scale_data.keys()]
            scale_scores = list(scale_data.values())
            
            ax2.semilogx(scales, scale_scores, 'o-', linewidth=2, markersize=8, color='red')
            ax2.set_title('Robustness to Reward Scaling')
            ax2.set_xlabel('Reward Scale')
            ax2.set_ylabel('Final Average Score')
            ax2.grid(True, alpha=0.3)
        
        # Performance distribution
        ax3 = axes[1, 0]
        all_scores = []
        if 'seed_robustness' in robustness_results:
            all_scores.extend(robustness_results['seed_robustness'].values())
        if 'scale_robustness' in robustness_results:
            all_scores.extend(robustness_results['scale_robustness'].values())
        
        if all_scores:
            ax3.hist(all_scores, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax3.set_title('Overall Performance Distribution')
            ax3.set_xlabel('Final Average Score')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Robustness statistics
        ax4 = axes[1, 1]
        if all_scores:
            stats = {
                'Mean': np.mean(all_scores),
                'Std': np.std(all_scores),
                'Min': np.min(all_scores),
                'Max': np.max(all_scores),
                'CV': np.std(all_scores) / np.mean(all_scores) * 100
            }
            
            stat_names = list(stats.keys())
            stat_values = list(stats.values())
            
            bars = ax4.bar(stat_names, stat_values, alpha=0.7, color='purple')
            ax4.set_title('Robustness Statistics')
            ax4.set_ylabel('Value')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, stat_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_report(self, results: Dict[str, Any], 
                            save_name: str = "summary_report.png"):
        """Create a comprehensive summary report"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Extract data
        if 'variants' in results:
            variants = results['variants']
            variant_names = list(variants.keys())
            variant_scores = [np.mean(v['scores'][-50:]) for v in variants.values()]
        else:
            variant_names = []
            variant_scores = []
        
        # 1. Best performing variant
        if variant_names:
            best_idx = np.argmax(variant_scores)
            best_variant = variant_names[best_idx]
            best_score = variant_scores[best_idx]
            
            axes[0, 0].text(0.5, 0.7, f'Best Variant: {best_variant}', 
                           ha='center', va='center', fontsize=16, fontweight='bold',
                           transform=axes[0, 0].transAxes)
            axes[0, 0].text(0.5, 0.5, f'Score: {best_score:.2f}', 
                           ha='center', va='center', fontsize=14,
                           transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Best Performance')
            axes[0, 0].axis('off')
        
        # 2. Performance comparison
        if variant_names:
            axes[0, 1].bar(variant_names, variant_scores, alpha=0.7, color='blue')
            axes[0, 1].set_title('Performance Comparison')
            axes[0, 1].set_ylabel('Final Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training stability
        if variant_names:
            stability_scores = [np.std(v['scores'][-100:]) for v in variants.values()]
            axes[0, 2].bar(variant_names, stability_scores, alpha=0.7, color='red')
            axes[0, 2].set_title('Training Stability')
            axes[0, 2].set_ylabel('Score Std Dev')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Learning curves
        if variant_names:
            colors = ['blue', 'green', 'red', 'purple']
            for i, (name, data) in enumerate(variants.items()):
                scores = data['scores']
                smoothed = np.convolve(scores, np.ones(20)/20, mode='valid')
                axes[1, 0].plot(smoothed, label=name, color=colors[i % len(colors)], linewidth=2)
            axes[1, 0].set_title('Learning Curves')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Smoothed Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Hyperparameter sensitivity
        if 'hyperparameters' in results:
            hyper_data = results['hyperparameters']
            if 'architectures' in hyper_data:
                arch_names = list(hyper_data['architectures'].keys())
                arch_scores = list(hyper_data['architectures'].values())
                axes[1, 1].bar(arch_names, arch_scores, alpha=0.7, color='green')
                axes[1, 1].set_title('Architecture Sensitivity')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Robustness analysis
        if 'robustness' in results:
            robust_data = results['robustness']
            if 'seed_robustness' in robust_data:
                seed_scores = list(robust_data['seed_robustness'].values())
                axes[1, 2].hist(seed_scores, bins=10, alpha=0.7, color='orange')
                axes[1, 2].set_title('Seed Robustness')
                axes[1, 2].set_xlabel('Score')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Key insights
        insights = [
            "Key Insights:",
            f"• Best variant: {best_variant if variant_names else 'N/A'}",
            "• Double DQN reduces overestimation bias",
            "• Dueling architecture improves value estimation",
            "• Experience replay stabilizes training",
            "• Target networks prevent divergence",
            "• Optimal learning rate: 1e-3",
            "• Robust to reward scaling"
        ]
        
        axes[2, 0].text(0.05, 0.95, '\n'.join(insights), 
                       ha='left', va='top', fontsize=12,
                       transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Key Insights')
        axes[2, 0].axis('off')
        
        # 8. Recommendations
        recommendations = [
            "Recommendations:",
            "• Use Double DQN for better stability",
            "• Implement Dueling architecture for efficiency",
            "• Start with learning rate 1e-3",
            "• Use experience replay buffer size 10000",
            "• Update target network every 10 steps",
            "• Monitor Q-value distributions",
            "• Test robustness across seeds"
        ]
        
        axes[2, 1].text(0.05, 0.95, '\n'.join(recommendations), 
                       ha='left', va='top', fontsize=12,
                       transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Recommendations')
        axes[2, 1].axis('off')
        
        # 9. Performance metrics
        if variant_names:
            metrics = {
                'Best Score': best_score,
                'Mean Score': np.mean(variant_scores),
                'Score Range': np.max(variant_scores) - np.min(variant_scores),
                'Stability': np.mean(stability_scores)
            }
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[2, 2].bar(metric_names, metric_values, alpha=0.7, color='purple')
            axes[2, 2].set_title('Performance Metrics')
            axes[2, 2].set_ylabel('Value')
            axes[2, 2].tick_params(axis='x', rotation=45)
            axes[2, 2].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                axes[2, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{value:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

class DQNAnalyzer:
    """Analysis tools for DQN performance and behavior"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_convergence(self, scores: List[float], target: float = 180, 
                          window: int = 20) -> Dict[str, Any]:
        """Analyze convergence behavior"""
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        
        # Find convergence point
        converged_idx = np.where(smoothed >= target)[0]
        convergence_episode = converged_idx[0] if len(converged_idx) > 0 else len(smoothed)
        
        # Calculate convergence metrics
        final_performance = np.mean(scores[-50:])
        stability = np.std(scores[-100:])
        
        return {
            'convergence_episode': convergence_episode,
            'final_performance': final_performance,
            'stability': stability,
            'target_reached': convergence_episode < len(smoothed),
            'smoothed_scores': smoothed.tolist()
        }
    
    def analyze_exploration_efficiency(self, epsilon_history: List[float], 
                                     scores: List[float]) -> Dict[str, Any]:
        """Analyze exploration efficiency"""
        # Find exploration phases
        high_exploration = np.array(epsilon_history) > 0.5
        medium_exploration = (np.array(epsilon_history) > 0.1) & (np.array(epsilon_history) <= 0.5)
        low_exploration = np.array(epsilon_history) <= 0.1
        
        # Calculate performance in each phase
        high_exp_scores = np.array(scores)[high_exploration]
        medium_exp_scores = np.array(scores)[medium_exploration]
        low_exp_scores = np.array(scores)[low_exploration]
        
        return {
            'high_exploration_performance': np.mean(high_exp_scores) if len(high_exp_scores) > 0 else 0,
            'medium_exploration_performance': np.mean(medium_exp_scores) if len(medium_exp_scores) > 0 else 0,
            'low_exploration_performance': np.mean(low_exp_scores) if len(low_exp_scores) > 0 else 0,
            'exploration_efficiency': np.mean(low_exp_scores) / np.mean(high_exp_scores) if len(high_exp_scores) > 0 and len(low_exp_scores) > 0 else 0
        }
    
    def compare_agents(self, agent_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare multiple agents"""
        comparison = {}
        
        for agent_name, results in agent_results.items():
            scores = results['scores']
            losses = results.get('losses', [])
            
            comparison[agent_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'final_score': np.mean(scores[-50:]),
                'convergence': self.analyze_convergence(scores),
                'mean_loss': np.mean(losses) if losses else 0,
                'training_stability': np.std(scores[-100:])
            }
        
        # Find best agent
        best_agent = max(comparison.keys(), key=lambda x: comparison[x]['final_score'])
        comparison['best_agent'] = best_agent
        
        return comparison
    
    def save_analysis(self, filename: str = "dqn_analysis.json"):
        """Save analysis results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

def create_environment_wrapper(env_name: str, **kwargs):
    """Create environment with optional wrappers"""
    env = gym.make(env_name, **kwargs)
    return env

def run_benchmark(agent_class, env_name: str = "CartPole-v1", 
                 episodes: int = 200, num_runs: int = 5) -> Dict[str, Any]:
    """Run benchmark with multiple random seeds"""
    results = []
    
    for run in range(num_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = agent_class(state_dim=state_dim, action_dim=action_dim)
        
        scores = []
        for episode in range(episodes):
            reward, _ = agent.train_episode(env, max_steps=500)
            scores.append(reward)
        
        results.append(scores)
        env.close()
    
    # Calculate statistics
    results_array = np.array(results)
    
    return {
        'mean_scores': np.mean(results_array, axis=0),
        'std_scores': np.std(results_array, axis=0),
        'final_mean': np.mean(results_array[:, -50:]),
        'final_std': np.std(results_array[:, -50:]),
        'individual_runs': results
    }