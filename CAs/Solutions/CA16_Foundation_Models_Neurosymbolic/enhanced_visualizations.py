"""
Enhanced Visualizations for CA16 Notebook

This module provides comprehensive visualization functions for:
- Attention mechanism analysis
- Training dynamics visualization
- Performance comparisons
- Interactive plots
- Uncertainty visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


def create_attention_heatmap(attention_weights, titles=None, figsize=(12, 8)):
    """
    Create comprehensive attention heatmap visualization.
    """
    if titles is None:
        titles = ['Head ' + str(i) for i in range(len(attention_weights))]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, (attn, title) in enumerate(zip(attention_weights[:4], titles[:4])):
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()
        
        # Average across batch if needed
        if attn.ndim == 3:
            attn = attn.mean(0)
        if attn.ndim == 1:
            attn = attn.reshape(1, -1)
            
        im = axes[i].imshow(attn, cmap='viridis', aspect='auto')
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
        
    plt.tight_layout()
    return fig


def plot_training_dynamics(losses, rewards=None, metrics=None, figsize=(15, 5)):
    """
    Comprehensive training dynamics visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss curves
    if losses:
        axes[0].plot(losses, 'b-', linewidth=2, label='Training Loss')
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Add moving average
        if len(losses) > 10:
            moving_avg = np.convolve(losses, np.ones(10)/10, mode='valid')
            axes[0].plot(range(9, len(losses)), moving_avg, 'r--', linewidth=2, label='Moving Avg (10)')
        
        axes[0].legend()
    
    # Reward curves
    if rewards:
        axes[1].plot(rewards, 'g-', linewidth=2, label='Episode Reward')
        axes[1].set_title('Episode Rewards', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel('Reward')
        axes[1].grid(True, alpha=0.3)
        
        # Add confidence interval
        if len(rewards) > 5:
            window = min(10, len(rewards)//5)
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(rewards[max(0, i-window):i+1]) for i in range(window-1, len(rewards))])
            
            axes[1].fill_between(range(window-1, len(rewards)), 
                               rolling_mean - rolling_std, 
                               rolling_mean + rolling_std, 
                               alpha=0.2, color='green', label=f'±1σ ({window}-ep avg)')
            axes[1].plot(range(window-1, len(rewards)), rolling_mean, 'g--', linewidth=2)
        
        axes[1].legend()
    
    # Additional metrics
    if metrics:
        metric_names = list(metrics.keys())
        for i, metric_name in enumerate(metric_names[:1]):  # Show one metric
            metric_values = metrics[metric_name]
            axes[2].plot(metric_values, 'purple', linewidth=2, marker='o', markersize=3, label=metric_name)
        
        axes[2].set_title('Additional Metrics', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Steps/Episodes')
        axes[2].set_ylabel('Metric Value')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    return fig


def create_performance_comparison(results_dict, figsize=(12, 8)):
    """
    Create comprehensive performance comparison visualization.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    methods = list(results_dict.keys())
    final_scores = [results_dict[method].get('final_score', 0) for method in methods]
    sample_efficiency = [results_dict[method].get('sample_efficiency', 0) for method in methods]
    
    # Bar chart comparison
    bars = ax1.bar(methods, final_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax1.set_title('Final Performance Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Final Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Sample efficiency comparison
    bars2 = ax2.bar(methods, sample_efficiency, color=['#A8E6CF', '#FFD93D', '#6BCFFC', '#FFB19A', '#C44569'])
    ax2.set_title('Sample Efficiency Comparison', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Efficiency Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars2, sample_efficiency):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Pareto front analysis
    ax3.scatter(sample_efficiency, final_scores, s=100, c='red', alpha=0.7, edgecolors='black')
    for i, method in enumerate(methods):
        ax3.annotate(method, (sample_efficiency[i], final_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax3.set_title('Efficiency vs Performance Pareto Front', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Sample Efficiency')
    ax3.set_ylabel('Final Performance')
    ax3.grid(True, alpha=0.3)
    
    # Learning curves comparison
    for method in methods:
        if 'learning_curve' in results_dict[method]:
            curve = results_dict[method]['learning_curve']
            ax4.plot(curve, label=method, linewidth=2, alpha=0.8)
    
    ax4.set_title('Learning Curves Comparison', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Steps/Episodes')
    ax4.set_ylabel('Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_feature_space(features_dict, labels=None, figsize=(12, 6)):
    """
    Visualize high-dimensional features in 2D space using t-SNE style visualization.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data
    feature_names = list(features_dict.keys())
    feature_arrays = []
    
    for name in feature_names:
        feat = features_dict[name]
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        feature_arrays.append(feat.flatten()[:1000])  # Limit sample size
    
    combined_features = np.vstack(feature_arrays)
    
    # PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_features)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    start_idx = 0
    
    for i, name in enumerate(feature_names):
        end_idx = start_idx + len(feature_arrays[i])
        ax1.scatter(pca_result[start_idx:end_idx, 0], 
                   pca_result[start_idx:end_idx, 1], 
                   c=colors[i % len(colors)], 
                   label=name, alpha=0.6, s=30)
        start_idx = end_idx
    
    ax1.set_title('PCA Feature Space Visualization', fontweight='bold', fontsize=14)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature correlation heatmap
    if len(feature_names) > 1:
        feature_matrix = np.array(feature_arrays).T[:50]  # Sample first 50 points
        correlation_matrix = np.corrcoef(feature_matrix.T)
        
        im = ax2.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.set_yticklabels(feature_names)
        
        # Add correlation values
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax2, fraction=0.046)
    
    plt.tight_layout()
    return fig


def create_model_architecture_diagram(figsize=(15, 10)):
    """
    Create architectural diagrams showing model components and data flow.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Decision Transformer Architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_title('Decision Transformer Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Components
    components = [
        (1, 6, 2, 1, 'State Embed'),
        (1, 4, 2, 1, 'Action Embed'),
        (1, 2, 2, 1, 'Return Embed'),
        (4, 4, 2, 1, 'Positional\nEncoding'),
        (6, 5, 2, 1, 'Transformer\nEncoder'),
        (6, 2, 2, 1, 'Action Head'),
        (2, 2, 1, 1, 'Output')
    ]
    
    colors = ['#FFE5B4', '#FFB5B5', '#B5D7FF', '#FFD700', '#98FB98', '#DDA0DD', '#FF6347']
    
    for (x, y, w, h, text), color in zip(components, colors):
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Add arrows
    arrows = [
        ((3, 6.5), (4, 5.5)),  # State -> PE
        ((3, 4.5), (4, 4.5)),  # Action -> PE
        ((3, 2.5), (4, 3.5)),  # Return -> PE
        ((6, 5.5), (6, 2.5)),  # Transformer -> Output
    ]
    
    for (start, end) in arrows:
        ax1.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax1.axis('off')
    
    # Neurosymbolic Architecture
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.set_title('Neurosymbolic RL Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Components
    components2 = [
        (1, 6, 2, 1, 'Neural\nPerception'),
        (1, 4, 2, 1, 'Symbolic\nReasoning'),
        (1, 2, 2, 1, 'Rule\nApplication'),
        (4, 5, 2, 1, 'Feature\nFusion'),
        (6, 4, 2, 1, 'Policy\nNetwork'),
        (6, 2, 2, 1, 'Value\nNetwork'),
        (5, 1, 2, 0.5, 'Output')
    ]
    
    colors2 = ['#87CEEB', '#FFA500', '#32CD32', '#FF69B4', '#9370DB', '#20B2AA', '#DC143C']
    
    for (x, y, w, h, text), color in zip(components2, colors2):
        rect = Rectangle((x, y), w: h, facecolor=color, edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # Add data flow arrows
    arrows2 = [
        ((3, 6.5), (4, 5.5)),  # Neural -> Fusion
        ((3, 4.5), (4, 4.5)),  # Symbolic -> Fusion
        ((3, 2.5), (4, 3.5)),  # Rules -> Fusion
        ((6, 4.5), (6, 2.5)),  # Fusion -> Outputs
    ]
    
    for (start, end) in arrows2:
        ax2.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def plot_uncertainty_analysis(uncertainties_dict, figsize=(12, 6)):
    """
    Visualize model uncertainty and confidence measures.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    methods = list(uncertainties_dict.keys())
    
    # Uncertainty evolution
    for method in methods:
        if 'epistemic' in uncertainties_dict[method]:
            epistemic = uncertainties_dict[method]['epistemic']
            ax1.plot(epistemic, label=f'{method} Epistemic', linewidth=2)
        
        if 'aleatoric' in uncertainties_dict[method]:
            aleatoric = uncertainties_dict[method]['aleatoric'] 
            ax2.plot(aleatoric, label=f'{method} Aleatoric', linewidth=2)
    ax1.set_title('Epistemic Uncertainty Evolution', fontweight='bold')
    ax1.set_ylabel('Epistemic Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Aleatoric Uncertainty Evolution', fontweight='bold')
    ax2.set_ylabel('Aleatoric Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confidence distributions
    for method in methods:
        if 'confidence' in uncertainties_dict[method]:
            confidence = uncertainties_dict[method]['confidence']
            ax3.hist(confidence, alpha=0.6, label=method, bins=20)
    
    ax3.set_title('Confidence Distribution', fontweight='bold')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Uncertainty vs Performance scatter
    for method in methods:
        if 'performance' in uncertainties_dict[method] and 'total_uncertainty' in uncertainties_dict[method]:
            perf = uncertainties_dict[method]['performance']
            uncert = uncertainties_dict[method]['total_uncertainty']
            ax4.scatter(uncert, perf, label=method, s=50, alpha=0.7)
    
    ax4.set_title('Uncertainty vs Performance', fontweight='bold')
    ax4.set_xlabel('Total Uncertainty')
    ax4.set_ylabel('Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Usage example and integration with notebook
def run_enhanced_visualizations():
    """
    Example function to demonstrate all visualization capabilities.
    """
    print("🎨 Enhanced Visualization Dashboard")
    print("=" * 50)
    
    # Mock data for demonstration
    losses = np.random.exponential(1.0, 50) * np.exp(-np.linspace(0, 3, 50))
    rewards = np.cumsum(np.random.normal(0, 0.5, 30)) + np.linspace(0, 5, 30)
    
    # Attention weights (mock)
    attention_weights = [np.random.rand(8, 8) for _ in range(4)]
    
    # Feature space
    features_dict = {
        'Neural Features': np.random.randn(100, 64),
        'Symbolic Features': np.random.randn(100, 32),
    }
    
    # Performance comparison
    results_dict = {
        'Foundation': {'final_score': 0.85, 'sample_efficiency': 0.90, 'learning_curve': rewards},
        'Neurosymbolic': {'final_score': 0.82, 'sample_efficiency': 0.85, 'learning_curve': rewards * 0.9},
        'Collaborative': {'final_score': 0.88, 'sample_efficiency': 0.75, 'learning_curve': rewards * 1.1},
        'Continual': {'final_score': 0.79, 'sample_efficiency': 0.95, 'learning_curve': rewards * 0.8},
    }
    
    # Uncertainty analysis
    uncertainties_dict = {
        'Foundation': {
            'epistemic': np.random.exponential(0.5, 30),
            'aleatoric': np.random.exponential(0.3, 30),
            'confidence': np.random.beta(3, 1, 100),
            'performance': rewards,
            'total_uncertainty': np.random.exponential(0.4, 30)
        },
        'Neurosymbolic': {
            'epistemic': np.random.exponential(0.4, 30),
            'aleatoric': np.random.exponential(0.2, 30),
            'confidence': np.random.beta(4, 1, 100),
            'performance': rewards * 0.9,
            'total_uncertainty': np.random.exponential(0.3, 30)
        }
    }
    
    print("Creating visualizations...")
    
    # Generate all visualizations
    fig1 = create_attention_heatmap(attention_headers=['Query', 'Key', 'Value', 'Output'])
    plt.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig2 = plot_training_dynamics(losses, rewards)
    plt.savefig('training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig3 = create_performance_comparison(results_dict)
    plt.savefig('perance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig4 = visualize_feature_space(features_dict)
    plt.savefig('feature_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig5 = create_model_architecture_diagram()
    plt.savefig('architecture_diagrams.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig6 = plot_uncertainty_analysis(uncertainties_dict)
    plt.savefig('uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ All visualizations completed!")
    return True
