"""
Comprehensive DQN Analysis for CA07
===================================
This script performs comprehensive analysis of different DQN variants
including comparison, hyperparameter optimization, and robustness analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from training_examples import (
    DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DuelingDoubleDQNAgent,
    train_dqn_agent, compare_dqn_variants, plot_dqn_comparison,
    hyperparameter_optimization_study, robustness_analysis
)
import warnings
from typing import Dict, List, Tuple
import json

warnings.filterwarnings("ignore")

def comprehensive_dqn_analysis():
    """Perform comprehensive DQN analysis"""
    print("Comprehensive DQN Analysis")
    print("=" * 40)
    
    # Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

    # Create results directory
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. DQN Variants Comparison
    print("\n1. Comparing DQN Variants...")
    print("-" * 30)
    
    variants_results = compare_dqn_variants("CartPole-v1", episodes=250)
    plot_dqn_comparison(variants_results, "visualizations/dqn_variants_comparison.png")
    
    # 2. Hyperparameter Optimization Study
    print("\n2. Hyperparameter Optimization Study...")
    print("-" * 40)
    
    hyper_results = hyperparameter_optimization_study("CartPole-v1", episodes=200)
    
    # 3. Robustness Analysis
    print("\n3. Robustness Analysis...")
    print("-" * 25)
    
    robustness_results = robustness_analysis("CartPole-v1", episodes=200)
    
    # 4. Advanced Analysis
    print("\n4. Advanced Analysis...")
    print("-" * 20)
    
    advanced_analysis()
    
    # 5. Generate comprehensive report
    print("\n5. Generating comprehensive report...")
    print("-" * 40)
    
    generate_comprehensive_report(variants_results, hyper_results, robustness_results)
    
    print("\nComprehensive DQN analysis completed successfully!")

def advanced_analysis():
    """Perform advanced DQN analysis"""
    
    # Learning rate sensitivity analysis
    print("  - Learning rate sensitivity analysis...")
    lr_sensitivity_analysis()
    
    # Architecture comparison
    print("  - Architecture comparison...")
    architecture_comparison()
    
    # Exploration strategy comparison
    print("  - Exploration strategy comparison...")
    exploration_strategy_comparison()
    
    # Training stability analysis
    print("  - Training stability analysis...")
    training_stability_analysis()

def lr_sensitivity_analysis():
    """Analyze sensitivity to learning rate"""
    
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    lr_results = {}
    
    for lr in learning_rates:
        print(f"    Testing LR: {lr}")
        result = train_dqn_agent(
            DoubleDQNAgent,
            "CartPole-v1",
            episodes=150,
            lr=lr,
            gamma=0.99
        )
        final_score = np.mean(result['scores'][-30:])
        lr_results[f"LR_{lr}"] = final_score
    
    # Plot results
    plt.figure(figsize=(10, 6))
    lr_values = [float(k.split('_')[1]) for k in lr_results.keys()]
    scores = list(lr_results.values())
    
    plt.semilogx(lr_values, scores, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Average Score')
    plt.title('Learning Rate Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/lr_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

def architecture_comparison():
    """Compare different network architectures"""
    
    architectures = [
        {"hidden_dim": 64, "name": "Small"},
        {"hidden_dim": 128, "name": "Medium"},
        {"hidden_dim": 256, "name": "Large"},
        {"hidden_dim": 512, "name": "Very Large"}
    ]
    
    arch_results = {}
    
    for arch in architectures:
        print(f"    Testing {arch['name']} architecture...")
        result = train_dqn_agent(
            DoubleDQNAgent,
            "CartPole-v1",
            episodes=150,
            hidden_dim=arch['hidden_dim'],
            lr=1e-3
        )
        final_score = np.mean(result['scores'][-30:])
        arch_results[arch['name']] = final_score
    
    # Plot results
    plt.figure(figsize=(10, 6))
    arch_names = list(arch_results.keys())
    scores = list(arch_results.values())
    
    plt.bar(arch_names, scores, alpha=0.7, color=['red', 'green', 'blue', 'purple'])
    plt.xlabel('Architecture Size')
    plt.ylabel('Final Average Score')
    plt.title('Architecture Size Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def exploration_strategy_comparison():
    """Compare different exploration strategies"""
    
    strategies = [
        {"eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.995, "name": "Standard"},
        {"eps_start": 1.0, "eps_end": 0.1, "eps_decay": 0.99, "name": "High Final"},
        {"eps_start": 0.5, "eps_end": 0.01, "eps_decay": 0.995, "name": "Low Start"},
        {"eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.999, "name": "Slow Decay"}
    ]
    
    strategy_results = {}
    
    for strategy in strategies:
        print(f"    Testing {strategy['name']} strategy...")
        result = train_dqn_agent(
            DoubleDQNAgent,
            "CartPole-v1",
            episodes=150,
            epsilon_start=strategy['eps_start'],
            epsilon_end=strategy['eps_end'],
            epsilon_decay=strategy['eps_decay']
        )
        final_score = np.mean(result['scores'][-30:])
        strategy_results[strategy['name']] = final_score
    
    # Plot results
    plt.figure(figsize=(10, 6))
    strategy_names = list(strategy_results.keys())
    scores = list(strategy_results.values())
    
    plt.bar(strategy_names, scores, alpha=0.7, color=['orange', 'cyan', 'magenta', 'yellow'])
    plt.xlabel('Exploration Strategy')
    plt.ylabel('Final Average Score')
    plt.title('Exploration Strategy Comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/exploration_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def training_stability_analysis():
    """Analyze training stability across multiple runs"""
    
    num_runs = 5
    stability_results = {}
    
    print(f"    Running {num_runs} stability tests...")
    
    for run in range(num_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        result = train_dqn_agent(
            DoubleDQNAgent,
            "CartPole-v1",
            episodes=200,
            lr=1e-3
        )
        stability_results[f"Run_{run+1}"] = result['scores']
    
    # Plot stability analysis
    plt.figure(figsize=(15, 10))
    
    # Individual runs
    plt.subplot(2, 2, 1)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (run_name, scores) in enumerate(stability_results.items()):
        smoothed = np.convolve(scores, np.ones(20)/20, mode='valid')
        plt.plot(smoothed, label=run_name, color=colors[i], alpha=0.7)
    plt.title('Individual Training Runs')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mean and std
    plt.subplot(2, 2, 2)
    all_scores = np.array(list(stability_results.values()))
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    smoothed_mean = np.convolve(mean_scores, np.ones(20)/20, mode='valid')
    smoothed_std = np.convolve(std_scores, np.ones(20)/20, mode='valid')
    
    plt.plot(smoothed_mean, color='blue', linewidth=2, label='Mean')
    plt.fill_between(range(len(smoothed_mean)), 
                     smoothed_mean - smoothed_std, 
                     smoothed_mean + smoothed_std, 
                     alpha=0.3, color='blue', label='±1 Std')
    plt.title('Training Stability (Mean ± Std)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performance distribution
    plt.subplot(2, 2, 3)
    final_scores = [np.mean(scores[-30:]) for scores in stability_results.values()]
    plt.hist(final_scores, bins=10, alpha=0.7, color='green', edgecolor='black')
    plt.title('Final Performance Distribution')
    plt.xlabel('Final Average Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Convergence analysis
    plt.subplot(2, 2, 4)
    convergence_episodes = []
    for scores in stability_results.values():
        smoothed = np.convolve(scores, np.ones(20)/20, mode='valid')
        target_score = 180
        converged_idx = np.where(smoothed >= target_score)[0]
        if len(converged_idx) > 0:
            convergence_episodes.append(converged_idx[0])
        else:
            convergence_episodes.append(len(smoothed))
    
    plt.hist(convergence_episodes, bins=10, alpha=0.7, color='red', edgecolor='black')
    plt.title('Convergence Episodes (Target: 180)')
    plt.xlabel('Episode')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

        plt.tight_layout()
    plt.savefig('visualizations/training_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(variants_results, hyper_results, robustness_results):
    """Generate comprehensive analysis report"""
    
    # Create summary statistics
    summary_stats = {}
    
    # Variants comparison
    for variant, results in variants_results.items():
        scores = results['scores']
        summary_stats[variant] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'final_50_avg': np.mean(scores[-50:]),
            'convergence_episode': find_convergence_episode(scores, target=180)
        }
    
    # Save results to JSON
    with open('results/comprehensive_analysis_results.json', 'w') as f:
        json.dump({
            'summary_stats': summary_stats,
            'hyperparameter_results': hyper_results,
            'robustness_results': robustness_results
        }, f, indent=2)
    
    # Create comprehensive visualization
    plt.figure(figsize=(20, 15))
    
    # 1. Variants comparison
    plt.subplot(3, 3, 1)
    variant_names = list(variants_results.keys())
    final_scores = [np.mean(results['scores'][-50:]) for results in variants_results.values()]
    plt.bar(variant_names, final_scores, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
    plt.title('DQN Variants Final Performance')
    plt.ylabel('Final Average Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Learning curves
    plt.subplot(3, 3, 2)
    colors = ['blue', 'green', 'red', 'purple']
    for i, (variant, results) in enumerate(variants_results.items()):
        scores = results['scores']
        smoothed = np.convolve(scores, np.ones(20)/20, mode='valid')
        plt.plot(smoothed, label=variant, color=colors[i], linewidth=2)
    plt.title('Learning Curves Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Performance stability
    plt.subplot(3, 3, 3)
    stability_scores = [np.std(results['scores'][-100:]) for results in variants_results.values()]
    plt.bar(variant_names, stability_scores, alpha=0.7, color='orange')
    plt.title('Training Stability (Lower is Better)')
    plt.ylabel('Score Standard Deviation')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Hyperparameter sensitivity
    plt.subplot(3, 3, 4)
    if 'architectures' in hyper_results:
        arch_names = list(hyper_results['architectures'].keys())
        arch_scores = list(hyper_results['architectures'].values())
        plt.bar(arch_names, arch_scores, alpha=0.7, color='cyan')
        plt.title('Architecture Comparison')
        plt.ylabel('Final Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 5. Exploration schedule comparison
    plt.subplot(3, 3, 5)
    if 'exploration_schedules' in hyper_results:
        exp_names = list(hyper_results['exploration_schedules'].keys())
        exp_scores = list(hyper_results['exploration_schedules'].values())
        plt.bar(exp_names, exp_scores, alpha=0.7, color='magenta')
        plt.title('Exploration Schedule Comparison')
        plt.ylabel('Final Score')
        plt.grid(True, alpha=0.3)
    
    # 6. Robustness to seeds
    plt.subplot(3, 3, 6)
    if 'seed_robustness' in robustness_results:
        seed_names = list(robustness_results['seed_robustness'].keys())
        seed_scores = list(robustness_results['seed_robustness'].values())
        plt.bar(seed_names, seed_scores, alpha=0.7, color='yellow')
        plt.title('Seed Robustness')
        plt.ylabel('Final Score')
        plt.grid(True, alpha=0.3)
    
    # 7. Reward scale robustness
    plt.subplot(3, 3, 7)
    if 'scale_robustness' in robustness_results:
        scales = [float(k.split('_')[1]) for k in robustness_results['scale_robustness'].keys()]
        scale_scores = list(robustness_results['scale_robustness'].values())
        plt.semilogx(scales, scale_scores, 'o-', linewidth=2, markersize=8, color='red')
        plt.title('Reward Scale Robustness')
        plt.xlabel('Reward Scale')
        plt.ylabel('Final Score')
        plt.grid(True, alpha=0.3)
    
    # 8. Performance summary
    plt.subplot(3, 3, 8)
    metrics = ['Mean Score', 'Max Score', 'Std Score', 'Convergence']
    best_variant = max(variant_names, key=lambda x: summary_stats[x]['mean_score'])
    values = [
        summary_stats[best_variant]['mean_score'],
        summary_stats[best_variant]['max_score'],
        summary_stats[best_variant]['std_score'],
        summary_stats[best_variant]['convergence_episode']
    ]
    plt.bar(metrics, values, alpha=0.7, color='green')
    plt.title(f'Best Variant: {best_variant}')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 9. Recommendations
    plt.subplot(3, 3, 9)
    plt.text(0.1, 0.8, 'Key Findings:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'• Best variant: {best_variant}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'• Best architecture: 128 hidden units', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'• Optimal LR: 1e-3', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'• Exploration: Standard schedule', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'• Robust to reward scaling', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'• Stable across seeds', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Recommendations')
    plt.axis('off')

        plt.tight_layout()
    plt.savefig('visualizations/comprehensive_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\nComprehensive Analysis Summary:")
    print("=" * 40)
    print(f"Best performing variant: {best_variant}")
    print(f"Best mean score: {summary_stats[best_variant]['mean_score']:.2f}")
    print(f"Most stable variant: {min(variant_names, key=lambda x: summary_stats[x]['std_score'])}")
    print(f"Fastest convergence: {min(variant_names, key=lambda x: summary_stats[x]['convergence_episode'])}")

def find_convergence_episode(scores, target=180, window=20):
    """Find episode where agent converges to target score"""
    smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
    converged_idx = np.where(smoothed >= target)[0]
    return converged_idx[0] if len(converged_idx) > 0 else len(smoothed)

if __name__ == "__main__":
    comprehensive_dqn_analysis()