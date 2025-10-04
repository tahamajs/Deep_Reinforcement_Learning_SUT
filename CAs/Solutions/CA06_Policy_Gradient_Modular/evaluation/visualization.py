"""
Visualization utilities for policy gradient methods
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


class TrainingPlotter:
    """Plot training progress and metrics"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self, results: Dict[str, List[float]], 
                           title: str = "Training Curves", 
                           xlabel: str = "Episode", 
                           ylabel: str = "Reward",
                           save_name: Optional[str] = None) -> None:
        """Plot training curves for multiple algorithms"""
        plt.figure(figsize=(12, 8))
        
        for algo_name, rewards in results.items():
            # Smooth the curve
            if len(rewards) > 50:
                window_size = max(10, len(rewards) // 50)
                smoothed = self._smooth_curve(rewards, window_size)
                plt.plot(smoothed, label=f"{algo_name} (smoothed)", alpha=0.8)
            else:
                plt.plot(rewards, label=algo_name, alpha=0.8)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_loss_curves(self, losses: Dict[str, List[float]], 
                        title: str = "Training Loss Curves",
                        save_name: Optional[str] = None) -> None:
        """Plot loss curves"""
        plt.figure(figsize=(12, 8))
        
        for algo_name, loss_list in losses.items():
            if loss_list:
                plt.plot(loss_list, label=algo_name, alpha=0.8)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Update Step", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_comparison(self, results: Dict[str, Dict], 
                                  metrics: List[str] = None,
                                  save_name: Optional[str] = None) -> None:
        """Plot performance comparison across algorithms"""
        if metrics is None:
            metrics = ['mean_final_performance', 'stability', 'improvement']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        algo_names = list(results.keys())
        
        for i, metric in enumerate(metrics):
            values = [results[algo][metric] for algo in algo_names]
            
            bars = axes[i].bar(algo_names, values, alpha=0.7)
            axes[i].set_title(f"{metric.replace('_', ' ').title()}", fontsize=14)
            axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence_analysis(self, results: Dict[str, List[float]], 
                                target_performance: Optional[float] = None,
                                save_name: Optional[str] = None) -> None:
        """Plot convergence analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Raw training curves
        for algo_name, rewards in results.items():
            axes[0, 0].plot(rewards, label=algo_name, alpha=0.7)
        axes[0, 0].set_title("Raw Training Curves", fontsize=14)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Smoothed curves
        for algo_name, rewards in results.items():
            smoothed = self._smooth_curve(rewards, max(10, len(rewards)//50))
            axes[0, 1].plot(smoothed, label=algo_name, linewidth=2)
        axes[0, 1].set_title("Smoothed Training Curves", fontsize=14)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if target_performance:
            axes[0, 1].axhline(y=target_performance, color='red', linestyle='--', 
                              alpha=0.7, label=f'Target: {target_performance:.1f}')
        
        # Plot 3: Performance distribution (final 20% of episodes)
        final_performances = {}
        for algo_name, rewards in results.items():
            final_portion = int(0.2 * len(rewards))
            final_performances[algo_name] = rewards[-final_portion:]
        
        axes[1, 0].boxplot(final_performances.values(), labels=final_performances.keys())
        axes[1, 0].set_title("Final Performance Distribution", fontsize=14)
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Sample efficiency (episodes to reach 80% of final performance)
        sample_efficiency = {}
        for algo_name, rewards in results.items():
            if len(rewards) > 100:
                final_perf = np.mean(rewards[-50:])
                target = 0.8 * final_perf
                for i, reward in enumerate(rewards):
                    if reward >= target:
                        sample_efficiency[algo_name] = i
                        break
        
        if sample_efficiency:
            algo_names = list(sample_efficiency.keys())
            episodes = list(sample_efficiency.values())
            axes[1, 1].bar(algo_names, episodes, alpha=0.7)
            axes[1, 1].set_title("Sample Efficiency (80% of Final)", fontsize=14)
            axes[1, 1].set_ylabel("Episodes")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _smooth_curve(self, data: List[float], window_size: int) -> List[float]:
        """Smooth curve using moving average"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            smoothed.append(np.mean(data[start_idx:i+1]))
        
        return smoothed


class PerformanceVisualizer:
    """Visualize performance metrics and statistics"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_performance_heatmap(self, results: Dict[str, Dict], 
                               save_name: Optional[str] = None) -> None:
        """Plot performance heatmap"""
        # Extract metrics
        metrics = ['mean_final_performance', 'stability', 'improvement', 'convergence_episode']
        algo_names = list(results.keys())
        
        # Create data matrix
        data_matrix = []
        for algo in algo_names:
            row = []
            for metric in metrics:
                if metric in results[algo]:
                    value = results[algo][metric]
                    # Normalize convergence episode (lower is better)
                    if metric == 'convergence_episode' and value is not None:
                        value = 1 / (1 + value / 1000)  # Normalize to 0-1
                    row.append(value)
                else:
                    row.append(0)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=algo_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   cbar_kws={'label': 'Normalized Performance'})
        
        plt.title("Algorithm Performance Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_statistical_comparison(self, results: Dict[str, List[float]], 
                                  save_name: Optional[str] = None) -> None:
        """Plot statistical comparison of algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract final performance data
        final_data = {}
        for algo_name, rewards in results.items():
            final_portion = int(0.2 * len(rewards))
            final_data[algo_name] = rewards[-final_portion:]
        
        # Plot 1: Violin plot
        data_for_violin = []
        labels_for_violin = []
        for algo_name, data in final_data.items():
            data_for_violin.extend(data)
            labels_for_violin.extend([algo_name] * len(data))
        
        axes[0, 0].violinplot([final_data[algo] for algo in final_data.keys()], 
                             positions=range(len(final_data.keys())))
        axes[0, 0].set_xticks(range(len(final_data.keys())))
        axes[0, 0].set_xticklabels(final_data.keys())
        axes[0, 0].set_title("Performance Distribution", fontsize=14)
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot
        axes[0, 1].boxplot([final_data[algo] for algo in final_data.keys()],
                          labels=list(final_data.keys()))
        axes[0, 1].set_title("Performance Quartiles", fontsize=14)
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mean and confidence intervals
        means = [np.mean(final_data[algo]) for algo in final_data.keys()]
        stds = [np.std(final_data[algo]) for algo in final_data.keys()]
        
        x_pos = range(len(final_data.keys()))
        axes[1, 0].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(final_data.keys())
        axes[1, 0].set_title("Mean Performance ± 1σ", fontsize=14)
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance over time (final episodes)
        for algo_name, rewards in results.items():
            final_episodes = int(0.3 * len(rewards))
            final_rewards = rewards[-final_episodes:]
            axes[1, 1].plot(final_rewards, label=algo_name, alpha=0.8)
        
        axes[1, 1].set_title("Final Training Phase", fontsize=14)
        axes[1, 1].set_xlabel("Episode (final 30%)")
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()


class PolicyVisualizer:
    """Visualize policy behavior and decision making"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_policy_heatmap(self, policy_fn, state_space: Tuple, 
                          action_space: int, title: str = "Policy Heatmap",
                          save_name: Optional[str] = None) -> None:
        """Plot policy as heatmap for 2D state spaces"""
        if len(state_space) != 2:
            print("Policy heatmap only supports 2D state spaces")
            return
        
        # Create state grid
        x_min, x_max = state_space[0]
        y_min, y_max = state_space[1]
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # Get policy for each state
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], Y[i, j]])
                with torch.no_grad():
                    action_probs = policy_fn(torch.FloatTensor(state).unsqueeze(0))
                    # Use the probability of the most likely action
                    Z[i, j] = torch.max(action_probs).item()
        
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Max Action Probability')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("State Dimension 1", fontsize=14)
        plt.ylabel("State Dimension 2", fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_action_distribution(self, policy_fn, states: List[np.ndarray], 
                               title: str = "Action Distribution",
                               save_name: Optional[str] = None) -> None:
        """Plot action distribution for given states"""
        all_actions = []
        
        for state in states:
            with torch.no_grad():
                action_probs = policy_fn(torch.FloatTensor(state).unsqueeze(0))
                action = torch.multinomial(action_probs, 1).item()
                all_actions.append(action)
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_actions, bins=range(max(all_actions) + 2), alpha=0.7, edgecolor='black')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Action", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
