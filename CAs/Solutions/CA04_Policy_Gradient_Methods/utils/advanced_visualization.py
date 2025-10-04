"""
Advanced Visualization and Analysis Tools
CA4: Policy Gradient Methods and Neural Networks in RL - Advanced Implementation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AdvancedPolicyVisualizer:
    """Advanced Policy Visualization Tools"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_policy_landscape(self, policy_net: nn.Module, state_space: np.ndarray, 
                            action_space: np.ndarray, resolution: int = 50):
        """Plot policy landscape over state space
        
        Args:
            policy_net: Policy network
            state_space: State space bounds
            action_space: Action space
            resolution: Resolution for grid
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Policy Landscape Analysis', fontsize=16, fontweight='bold')
        
        # Create state grid
        if len(state_space) == 2:
            x = np.linspace(state_space[0][0], state_space[0][1], resolution)
            y = np.linspace(state_space[1][0], state_space[1][1], resolution)
            X, Y = np.meshgrid(x, y)
            
            # Get policy probabilities
            states = np.stack([X.ravel(), Y.ravel()], axis=1)
            states_tensor = torch.FloatTensor(states)
            
            with torch.no_grad():
                logits = policy_net(states_tensor)
                probs = F.softmax(logits, dim=-1)
                
                # Plot action probabilities
                for action_idx in range(min(4, probs.shape[1])):
                    ax = axes[action_idx // 2, action_idx % 2]
                    Z = probs[:, action_idx].numpy().reshape(X.shape)
                    
                    im = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
                    ax.set_title(f'Action {action_idx} Probability', fontweight='bold')
                    ax.set_xlabel('State Dimension 1')
                    ax.set_ylabel('State Dimension 2')
                    plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('visualizations/policy_landscape.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_policy_entropy(self, policy_net: nn.Module, state_space: np.ndarray, 
                           resolution: int = 50):
        """Plot policy entropy over state space
        
        Args:
            policy_net: Policy network
            state_space: State space bounds
            resolution: Resolution for grid
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(state_space) == 2:
            x = np.linspace(state_space[0][0], state_space[0][1], resolution)
            y = np.linspace(state_space[1][0], state_space[1][1], resolution)
            X, Y = np.meshgrid(x, y)
            
            states = np.stack([X.ravel(), Y.ravel()], axis=1)
            states_tensor = torch.FloatTensor(states)
            
            with torch.no_grad():
                logits = policy_net(states_tensor)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                Z = entropy.numpy().reshape(X.shape)
                
                im = ax.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.8)
                ax.set_title('Policy Entropy Landscape', fontsize=14, fontweight='bold')
                ax.set_xlabel('State Dimension 1')
                ax.set_ylabel('State Dimension 2')
                plt.colorbar(im, ax=ax, label='Entropy')
        
        plt.tight_layout()
        plt.savefig('visualizations/policy_entropy.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_value_function(self, value_net: nn.Module, state_space: np.ndarray, 
                           resolution: int = 50):
        """Plot value function over state space
        
        Args:
            value_net: Value network
            state_space: State space bounds
            resolution: Resolution for grid
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(state_space) == 2:
            x = np.linspace(state_space[0][0], state_space[0][1], resolution)
            y = np.linspace(state_space[1][0], state_space[1][1], resolution)
            X, Y = np.meshgrid(x, y)
            
            states = np.stack([X.ravel(), Y.ravel()], axis=1)
            states_tensor = torch.FloatTensor(states)
            
            with torch.no_grad():
                values = value_net(states_tensor).squeeze()
                Z = values.numpy().reshape(X.shape)
                
                im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu', alpha=0.8)
                ax.set_title('Value Function Landscape', fontsize=14, fontweight='bold')
                ax.set_xlabel('State Dimension 1')
                ax.set_ylabel('State Dimension 2')
                plt.colorbar(im, ax=ax, label='Value')
        
        plt.tight_layout()
        plt.savefig('visualizations/value_function.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_advantage_function(self, advantages: List[torch.Tensor], states: List[np.ndarray]):
        """Plot advantage function
        
        Args:
            advantages: List of advantage values
            states: List of states
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to numpy arrays
        adv_array = np.array([adv.item() for adv in advantages])
        states_array = np.array(states)
        
        if states_array.shape[1] == 1:
            ax.scatter(states_array.flatten(), adv_array, alpha=0.6, s=20)
            ax.set_xlabel('State')
            ax.set_ylabel('Advantage')
            ax.set_title('Advantage Function', fontsize=14, fontweight='bold')
        elif states_array.shape[1] == 2:
            scatter = ax.scatter(states_array[:, 0], states_array[:, 1], 
                               c=adv_array, cmap='RdYlBu', alpha=0.7, s=30)
            ax.set_xlabel('State Dimension 1')
            ax.set_ylabel('State Dimension 2')
            ax.set_title('Advantage Function', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Advantage')
        
        plt.tight_layout()
        plt.savefig('visualizations/advantage_function.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_policy_gradients(self, gradients: List[torch.Tensor], layer_names: List[str]):
        """Plot policy gradients
        
        Args:
            gradients: List of gradient tensors
            layer_names: Names of layers
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Policy Gradient Analysis', fontsize=16, fontweight='bold')
        
        for i, (grad, name) in enumerate(zip(gradients, layer_names)):
            if i >= 4:
                break
                
            ax = axes[i // 2, i % 2]
            
            # Flatten gradients
            grad_flat = grad.view(-1).detach().numpy()
            
            # Plot histogram
            ax.hist(grad_flat, bins=50, alpha=0.7, color=self.colors[i % len(self.colors)])
            ax.set_title(f'{name} Gradients', fontweight='bold')
            ax.set_xlabel('Gradient Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/policy_gradients.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_network_architecture(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """Plot network architecture
        
        Args:
            model: Neural network model
            input_shape: Input shape
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes for layers
        layer_idx = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                G.add_node(layer_idx, label=name, type=type(module).__name__)
                layer_idx += 1
        
        # Add edges
        for i in range(layer_idx - 1):
            G.add_edge(i, i + 1)
        
        # Position nodes
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_colors = [self.colors[i % len(self.colors)] for i in range(layer_idx)]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Draw labels
        labels = {i: G.nodes[i]['type'] for i in range(layer_idx)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        ax.set_title('Network Architecture', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/network_architecture.png', dpi=300, bbox_inches='tight')
        plt.show()


class AdvancedTrainingVisualizer:
    """Advanced Training Visualization Tools"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_training_metrics(self, metrics: Dict[str, List[float]], 
                            window_size: int = 100):
        """Plot comprehensive training metrics
        
        Args:
            metrics: Dictionary of metrics
            window_size: Window size for smoothing
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Metrics Analysis', fontsize=16, fontweight='bold')
        
        # Plot rewards
        if 'scores' in metrics:
            ax = axes[0, 0]
            scores = metrics['scores']
            smoothed_scores = self._smooth_curve(scores, window_size)
            
            ax.plot(scores, alpha=0.3, color='lightblue', label='Raw')
            ax.plot(smoothed_scores, color='blue', linewidth=2, label='Smoothed')
            ax.set_title('Episode Rewards', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot losses
        if 'policy_losses' in metrics:
            ax = axes[0, 1]
            losses = metrics['policy_losses']
            smoothed_losses = self._smooth_curve(losses, window_size)
            
            ax.plot(losses, alpha=0.3, color='lightcoral', label='Raw')
            ax.plot(smoothed_losses, color='red', linewidth=2, label='Smoothed')
            ax.set_title('Policy Losses', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot value losses
        if 'value_losses' in metrics:
            ax = axes[0, 2]
            losses = metrics['value_losses']
            smoothed_losses = self._smooth_curve(losses, window_size)
            
            ax.plot(losses, alpha=0.3, color='lightgreen', label='Raw')
            ax.plot(smoothed_losses, color='green', linewidth=2, label='Smoothed')
            ax.set_title('Value Losses', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot entropy
        if 'entropy_losses' in metrics:
            ax = axes[1, 0]
            losses = metrics['entropy_losses']
            smoothed_losses = self._smooth_curve(losses, window_size)
            
            ax.plot(losses, alpha=0.3, color='lightyellow', label='Raw')
            ax.plot(smoothed_losses, color='orange', linewidth=2, label='Smoothed')
            ax.set_title('Entropy Losses', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot KL divergence
        if 'kl_divergences' in metrics:
            ax = axes[1, 1]
            kl_divs = metrics['kl_divergences']
            smoothed_kl = self._smooth_curve(kl_divs, window_size)
            
            ax.plot(kl_divs, alpha=0.3, color='lightpink', label='Raw')
            ax.plot(smoothed_kl, color='purple', linewidth=2, label='Smoothed')
            ax.set_title('KL Divergences', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('KL Divergence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot TD errors
        if 'td_errors' in metrics:
            ax = axes[1, 2]
            td_errors = metrics['td_errors']
            smoothed_td = self._smooth_curve(td_errors, window_size)
            
            ax.plot(td_errors, alpha=0.3, color='lightcyan', label='Raw')
            ax.plot(smoothed_td, color='cyan', linewidth=2, label='Smoothed')
            ax.set_title('TD Errors', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('TD Error')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves_comparison(self, results: Dict[str, Dict[str, List[float]]], 
                                      window_size: int = 100):
        """Plot learning curves comparison
        
        Args:
            results: Dictionary of algorithm results
            window_size: Window size for smoothing
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
        
        # Plot rewards comparison
        ax = axes[0, 0]
        for i, (alg_name, alg_results) in enumerate(results.items()):
            if 'scores' in alg_results:
                scores = alg_results['scores']
                smoothed_scores = self._smooth_curve(scores, window_size)
                ax.plot(smoothed_scores, label=alg_name, color=self.colors[i % len(self.colors)], 
                       linewidth=2)
        
        ax.set_title('Episode Rewards Comparison', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot policy losses comparison
        ax = axes[0, 1]
        for i, (alg_name, alg_results) in enumerate(results.items()):
            if 'policy_losses' in alg_results:
                losses = alg_results['policy_losses']
                smoothed_losses = self._smooth_curve(losses, window_size)
                ax.plot(smoothed_losses, label=alg_name, color=self.colors[i % len(self.colors)], 
                       linewidth=2)
        
        ax.set_title('Policy Losses Comparison', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot value losses comparison
        ax = axes[1, 0]
        for i, (alg_name, alg_results) in enumerate(results.items()):
            if 'value_losses' in alg_results:
                losses = alg_results['value_losses']
                smoothed_losses = self._smooth_curve(losses, window_size)
                ax.plot(smoothed_losses, label=alg_name, color=self.colors[i % len(self.colors)], 
                       linewidth=2)
        
        ax.set_title('Value Losses Comparison', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot convergence analysis
        ax = axes[1, 1]
        for i, (alg_name, alg_results) in enumerate(results.items()):
            if 'scores' in alg_results:
                scores = alg_results['scores']
                # Calculate moving average
                window = min(100, len(scores) // 10)
                if window > 0:
                    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
                    ax.plot(moving_avg, label=alg_name, color=self.colors[i % len(self.colors)], 
                           linewidth=2)
        
        ax.set_title('Convergence Analysis', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Moving Average Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_distribution(self, results: Dict[str, Dict[str, List[float]]]):
        """Plot performance distribution
        
        Args:
            results: Dictionary of algorithm results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Extract final performance
        final_scores = {}
        for alg_name, alg_results in results.items():
            if 'scores' in alg_results:
                scores = alg_results['scores']
                # Use last 10% of episodes
                final_scores[alg_name] = scores[-len(scores)//10:]
        
        # Box plot
        ax = axes[0, 0]
        data = [final_scores[alg] for alg in final_scores.keys()]
        labels = list(final_scores.keys())
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], self.colors[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Final Performance Distribution', fontweight='bold')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        # Violin plot
        ax = axes[0, 1]
        data_for_violin = []
        labels_for_violin = []
        for alg_name, scores in final_scores.items():
            data_for_violin.extend(scores)
            labels_for_violin.extend([alg_name] * len(scores))
        
        df = pd.DataFrame({'Algorithm': labels_for_violin, 'Reward': data_for_violin})
        sns.violinplot(data=df, x='Algorithm', y='Reward', ax=ax)
        ax.set_title('Performance Distribution (Violin)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Histogram comparison
        ax = axes[1, 0]
        for i, (alg_name, scores) in enumerate(final_scores.items()):
            ax.hist(scores, alpha=0.6, label=alg_name, color=self.colors[i % len(self.colors)], 
                   bins=20)
        
        ax.set_title('Performance Histogram', fontweight='bold')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistical summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for alg_name, scores in final_scores.items():
            summary_data.append([
                alg_name,
                f"{np.mean(scores):.2f}",
                f"{np.std(scores):.2f}",
                f"{np.median(scores):.2f}",
                f"{np.percentile(scores, 25):.2f}",
                f"{np.percentile(scores, 75):.2f}"
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Algorithm', 'Mean', 'Std', 'Median', 'Q1', 'Q3'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Statistical Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hyperparameter_sensitivity(self, results: Dict[str, Dict[str, Any]]):
        """Plot hyperparameter sensitivity analysis
        
        Args:
            results: Dictionary of hyperparameter results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # Extract hyperparameters and performance
        hyperparams = list(results.keys())
        performances = [results[hp]['final_performance'] for hp in hyperparams]
        
        # Plot 1: Bar chart of performance
        ax = axes[0, 0]
        bars = ax.bar(range(len(hyperparams)), performances, 
                     color=self.colors[:len(hyperparams)], alpha=0.7)
        ax.set_title('Performance by Hyperparameter', fontweight='bold')
        ax.set_xlabel('Hyperparameter Setting')
        ax.set_ylabel('Final Performance')
        ax.set_xticks(range(len(hyperparams)))
        ax.set_xticklabels(hyperparams, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, perf in zip(bars, performances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{perf:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Learning curves for different hyperparameters
        ax = axes[0, 1]
        for i, (hp_name, hp_results) in enumerate(results.items()):
            if 'scores' in hp_results:
                scores = hp_results['scores']
                smoothed_scores = self._smooth_curve(scores, 50)
                ax.plot(smoothed_scores, label=hp_name, color=self.colors[i % len(self.colors)], 
                       linewidth=2)
        
        ax.set_title('Learning Curves Comparison', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Performance vs hyperparameter value (if numeric)
        ax = axes[1, 0]
        try:
            # Try to extract numeric values
            hp_values = []
            for hp_name in hyperparams:
                # Extract numeric part from hyperparameter name
                import re
                numbers = re.findall(r'\d+\.?\d*', hp_name)
                if numbers:
                    hp_values.append(float(numbers[0]))
                else:
                    hp_values.append(len(hp_values))
            
            ax.scatter(hp_values, performances, s=100, alpha=0.7, 
                      c=self.colors[:len(hp_values)])
            ax.set_title('Performance vs Hyperparameter Value', fontweight='bold')
            ax.set_xlabel('Hyperparameter Value')
            ax.set_ylabel('Final Performance')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(hp_values) > 1:
                z = np.polyfit(hp_values, performances, 1)
                p = np.poly1d(z)
                ax.plot(hp_values, p(hp_values), "r--", alpha=0.8, linewidth=2)
        except:
            ax.text(0.5, 0.5, 'Non-numeric hyperparameters', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance vs Hyperparameter Value', fontweight='bold')
        
        # Plot 4: Stability analysis
        ax = axes[1, 1]
        stability_scores = []
        for hp_name, hp_results in results.items():
            if 'scores' in hp_results:
                scores = hp_results['scores']
                # Calculate stability as inverse of standard deviation
                stability = 1.0 / (np.std(scores) + 1e-8)
                stability_scores.append(stability)
            else:
                stability_scores.append(0)
        
        bars = ax.bar(range(len(hyperparams)), stability_scores, 
                     color=self.colors[:len(hyperparams)], alpha=0.7)
        ax.set_title('Training Stability', fontweight='bold')
        ax.set_xlabel('Hyperparameter Setting')
        ax.set_ylabel('Stability (1/std)')
        ax.set_xticks(range(len(hyperparams)))
        ax.set_xticklabels(hyperparams, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _smooth_curve(self, data: List[float], window_size: int) -> List[float]:
        """Smooth curve using moving average
        
        Args:
            data: Input data
            window_size: Window size
            
        Returns:
            Smoothed data
        """
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed


class AdvancedAnalysisTools:
    """Advanced Analysis Tools for Policy Gradient Methods"""
    
    def __init__(self):
        """Initialize analysis tools"""
        pass
    
    def analyze_policy_convergence(self, scores: List[float], window_size: int = 100) -> Dict[str, Any]:
        """Analyze policy convergence
        
        Args:
            scores: List of episode scores
            window_size: Window size for analysis
            
        Returns:
            Convergence analysis results
        """
        if len(scores) < window_size:
            return {'converged': False, 'convergence_episode': None, 'stability': 0}
        
        # Calculate moving average
        moving_avg = []
        for i in range(len(scores)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(scores), i + window_size // 2 + 1)
            moving_avg.append(np.mean(scores[start_idx:end_idx]))
        
        # Check for convergence
        convergence_threshold = 0.01
        converged = False
        convergence_episode = None
        
        for i in range(window_size, len(moving_avg)):
            recent_avg = np.mean(moving_avg[i-window_size:i])
            current_avg = moving_avg[i]
            
            if abs(current_avg - recent_avg) < convergence_threshold:
                converged = True
                convergence_episode = i
                break
        
        # Calculate stability
        final_scores = scores[-window_size:]
        stability = 1.0 / (np.std(final_scores) + 1e-8)
        
        return {
            'converged': converged,
            'convergence_episode': convergence_episode,
            'stability': stability,
            'final_performance': np.mean(final_scores),
            'performance_std': np.std(final_scores)
        }
    
    def analyze_gradient_flow(self, gradients: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze gradient flow
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Gradient flow analysis
        """
        if not gradients:
            return {'gradient_norm': 0, 'gradient_std': 0, 'vanishing_gradients': False}
        
        # Calculate gradient norms
        grad_norms = [torch.norm(grad).item() for grad in gradients if grad is not None]
        
        if not grad_norms:
            return {'gradient_norm': 0, 'gradient_std': 0, 'vanishing_gradients': False}
        
        avg_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)
        
        # Check for vanishing gradients
        vanishing_threshold = 1e-6
        vanishing_gradients = avg_norm < vanishing_threshold
        
        return {
            'gradient_norm': avg_norm,
            'gradient_std': std_norm,
            'vanishing_gradients': vanishing_gradients,
            'gradient_norms': grad_norms
        }
    
    def analyze_exploration_efficiency(self, actions: List[int], states: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze exploration efficiency
        
        Args:
            actions: List of actions taken
            states: List of states visited
            
        Returns:
            Exploration efficiency analysis
        """
        if not actions or not states:
            return {'entropy': 0, 'coverage': 0, 'efficiency': 0}
        
        # Calculate action entropy
        action_counts = np.bincount(actions)
        action_probs = action_counts / np.sum(action_counts)
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        
        # Calculate state coverage
        states_array = np.array(states)
        if states_array.shape[1] == 1:
            unique_states = len(np.unique(states_array))
            coverage = unique_states / len(states)
        else:
            # Use PCA for high-dimensional states
            pca = PCA(n_components=2)
            states_2d = pca.fit_transform(states_array)
            unique_states = len(np.unique(states_2d, axis=0))
            coverage = unique_states / len(states)
        
        # Calculate exploration efficiency
        efficiency = action_entropy * coverage
        
        return {
            'entropy': action_entropy,
            'coverage': coverage,
            'efficiency': efficiency,
            'unique_states': unique_states,
            'total_states': len(states)
        }
    
    def create_performance_report(self, results: Dict[str, Any]) -> str:
        """Create comprehensive performance report
        
        Args:
            results: Training results
            
        Returns:
            Performance report string
        """
        report = "=" * 60 + "\n"
        report += "POLICY GRADIENT TRAINING REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Basic statistics
        if 'scores' in results:
            scores = results['scores']
            report += f"Total Episodes: {len(scores)}\n"
            report += f"Final Performance: {np.mean(scores[-100:]):.2f} ± {np.std(scores[-100:]):.2f}\n"
            report += f"Best Performance: {np.max(scores):.2f}\n"
            report += f"Worst Performance: {np.min(scores):.2f}\n\n"
        
        # Loss analysis
        if 'policy_losses' in results:
            losses = results['policy_losses']
            report += f"Average Policy Loss: {np.mean(losses):.4f}\n"
            report += f"Final Policy Loss: {np.mean(losses[-100:]):.4f}\n\n"
        
        if 'value_losses' in results:
            losses = results['value_losses']
            report += f"Average Value Loss: {np.mean(losses):.4f}\n"
            report += f"Final Value Loss: {np.mean(losses[-100:]):.4f}\n\n"
        
        # Convergence analysis
        if 'scores' in results:
            conv_analysis = self.analyze_policy_convergence(results['scores'])
            report += "CONVERGENCE ANALYSIS:\n"
            report += f"Converged: {conv_analysis['converged']}\n"
            if conv_analysis['convergence_episode']:
                report += f"Convergence Episode: {conv_analysis['convergence_episode']}\n"
            report += f"Stability: {conv_analysis['stability']:.4f}\n\n"
        
        # Gradient analysis
        if 'gradients' in results:
            grad_analysis = self.analyze_gradient_flow(results['gradients'])
            report += "GRADIENT ANALYSIS:\n"
            report += f"Average Gradient Norm: {grad_analysis['gradient_norm']:.6f}\n"
            report += f"Gradient Std: {grad_analysis['gradient_std']:.6f}\n"
            report += f"Vanishing Gradients: {grad_analysis['vanishing_gradients']}\n\n"
        
        # Exploration analysis
        if 'actions' in results and 'states' in results:
            exp_analysis = self.analyze_exploration_efficiency(results['actions'], results['states'])
            report += "EXPLORATION ANALYSIS:\n"
            report += f"Action Entropy: {exp_analysis['entropy']:.4f}\n"
            report += f"State Coverage: {exp_analysis['coverage']:.4f}\n"
            report += f"Exploration Efficiency: {exp_analysis['efficiency']:.4f}\n\n"
        
        report += "=" * 60 + "\n"
        
        return report


def create_interactive_visualization(results: Dict[str, Any]) -> go.Figure:
    """Create interactive visualization using Plotly
    
    Args:
        results: Training results
        
    Returns:
        Plotly figure
    """
    if 'scores' not in results:
        return go.Figure()
    
    scores = results['scores']
    episodes = list(range(len(scores)))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Policy Losses', 'Value Losses', 'Training Progress'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=scores, mode='lines', name='Rewards', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Policy losses
    if 'policy_losses' in results:
        fig.add_trace(
            go.Scatter(x=episodes, y=results['policy_losses'], mode='lines', 
                      name='Policy Loss', line=dict(color='red')),
            row=1, col=2
        )
    
    # Value losses
    if 'value_losses' in results:
        fig.add_trace(
            go.Scatter(x=episodes, y=results['value_losses'], mode='lines', 
                      name='Value Loss', line=dict(color='green')),
            row=2, col=1
        )
    
    # Training progress (moving average)
    window_size = min(100, len(scores) // 10)
    if window_size > 0:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = list(range(window_size-1, len(scores)))
        fig.add_trace(
            go.Scatter(x=moving_episodes, y=moving_avg, mode='lines', 
                      name='Moving Average', line=dict(color='purple')),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Interactive Training Visualization",
        showlegend=True,
        height=800,
        width=1200
    )
    
    return fig

