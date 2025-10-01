"""
Visualization utilities for RL experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_learning_curves(results_dict, window=10, figsize=(12, 6)):
    """
    Plot learning curves for multiple agents.
    
    Args:
        results_dict: Dictionary mapping agent names to results
        window: Window size for moving average
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot rewards
    ax = axes[0]
    for agent_name, results in results_dict.items():
        rewards = results['rewards']
        smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, label=agent_name, linewidth=2)
        ax.fill_between(range(len(rewards)), 
                        pd.Series(rewards).rolling(window=window, min_periods=1).min(),
                        pd.Series(rewards).rolling(window=window, min_periods=1).max(),
                        alpha=0.2)
    
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot episode lengths
    ax = axes[1]
    for agent_name, results in results_dict.items():
        if 'lengths' in results:
            lengths = results['lengths']
            smoothed = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
            ax.plot(smoothed, label=agent_name, linewidth=2)
    
    ax.set_title('Episode Length Progression', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sample_efficiency(results_dict, figsize=(10, 6)):
    """
    Plot sample efficiency comparison.
    
    Args:
        results_dict: Dictionary mapping agent names to results
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    agent_names = list(results_dict.keys())
    convergence_episodes = []
    std_episodes = []
    
    for agent_name in agent_names:
        if 'sample_efficiency' in results_dict[agent_name]:
            eff = results_dict[agent_name]['sample_efficiency']
            convergence_episodes.append(eff.get('convergence_episodes', 0))
            std_episodes.append(eff.get('convergence_std', 0))
    
    x = np.arange(len(agent_names))
    bars = ax.bar(x, convergence_episodes, yerr=std_episodes, capsize=5, alpha=0.7)
    
    # Color bars based on performance (lower is better)
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title('Sample Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episodes to Convergence')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_loss_dynamics(results_dict, max_steps=1000, figsize=(10, 6)):
    """
    Plot training loss dynamics for multiple agents.
    
    Args:
        results_dict: Dictionary mapping agent names to results
        max_steps: Maximum number of steps to plot
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for agent_name, results in results_dict.items():
        if 'losses' in results:
            losses = results['losses'][:max_steps]
            ax.plot(losses, label=agent_name, alpha=0.7, linewidth=1.5)
    
    ax.set_title('Training Loss Dynamics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_performance_comparison(results_dict, figsize=(12, 6)):
    """
    Create comprehensive performance comparison visualization.
    
    Args:
        results_dict: Dictionary mapping agent names to results
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    agent_names = list(results_dict.keys())
    
    # Training performance
    ax = axes[0]
    final_returns = []
    for agent_name in agent_names:
        if 'rewards' in results_dict[agent_name]:
            final_returns.append(np.mean(results_dict[agent_name]['rewards'][-20:]))
        else:
            final_returns.append(0)
    
    bars = ax.barh(agent_names, final_returns, alpha=0.7)
    ax.set_title('Final Training Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Return (last 20 episodes)')
    ax.grid(alpha=0.3, axis='x')
    
    # Evaluation performance
    ax = axes[1]
    eval_returns = []
    eval_stds = []
    for agent_name in agent_names:
        if 'evaluation' in results_dict[agent_name]:
            eval_returns.append(results_dict[agent_name]['evaluation']['mean_return'])
            eval_stds.append(results_dict[agent_name]['evaluation']['std_return'])
        else:
            eval_returns.append(0)
            eval_stds.append(0)
    
    bars = ax.barh(agent_names, eval_returns, xerr=eval_stds, capsize=5, alpha=0.7)
    ax.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mean Return')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


__all__ = [
    'plot_learning_curves',
    'plot_sample_efficiency',
    'plot_loss_dynamics',
    'plot_performance_comparison'
]
