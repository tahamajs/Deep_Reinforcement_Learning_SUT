"""
Evaluation utilities for CA1 Deep RL Fundamentals.

This module contains functions for evaluating RL agents and comparing
their performance across different metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import json
import os


class AgentEvaluator:
    """
    A comprehensive evaluator for RL agents that can measure various
    performance metrics and generate detailed reports.
    """
    
    def __init__(self, agent, env, n_eval_episodes: int = 100):
        self.agent = agent
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.evaluation_results = {}
        
    def evaluate(self, render: bool = False) -> Dict[str, Any]:
        """
        Evaluate the agent over multiple episodes.
        
        Args:
            render: Whether to render the environment during evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        scores = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            score, length = self._run_episode(render=render)
            scores.append(score)
            episode_lengths.append(length)
            
            if (episode + 1) % 10 == 0:
                print(f"Evaluation episode {episode + 1}/{self.n_eval_episodes}")
        
        self.evaluation_results = {
            'scores': scores,
            'episode_lengths': episode_lengths,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'success_rate': np.mean([s >= 195.0 for s in scores])  # CartPole threshold
        }
        
        return self.evaluation_results
    
    def _run_episode(self, render: bool = False) -> Tuple[float, int]:
        """Run a single evaluation episode."""
        state, _ = self.env.reset()
        score = 0.0
        length = 0
        
        while True:
            if render:
                self.env.render()
                
            action = self.agent.act(state, eps=0.0)  # No exploration during evaluation
            
            result = self.env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
                
            state = next_state
            score += reward
            length += 1
            
            if done:
                break
                
        return score, length
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot evaluation results."""
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate() first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score distribution
        axes[0, 0].hist(self.evaluation_results['scores'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(self.evaluation_results['mean_score'], color='red', linestyle='--', 
                          label=f"Mean: {self.evaluation_results['mean_score']:.2f}")
        axes[0, 0].set_xlabel('Episode Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode length distribution
        axes[0, 1].hist(self.evaluation_results['episode_lengths'], bins=20, alpha=0.7, color='lightcoral')
        axes[0, 1].axvline(self.evaluation_results['mean_length'], color='red', linestyle='--',
                          label=f"Mean: {self.evaluation_results['mean_length']:.2f}")
        axes[0, 1].set_xlabel('Episode Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Episode Length Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Score vs Episode Length scatter
        axes[1, 0].scatter(self.evaluation_results['episode_lengths'], 
                          self.evaluation_results['scores'], alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Episode Length')
        axes[1, 0].set_ylabel('Episode Score')
        axes[1, 0].set_title('Score vs Episode Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics summary
        metrics = [
            f"Mean Score: {self.evaluation_results['mean_score']:.2f} ± {self.evaluation_results['std_score']:.2f}",
            f"Max Score: {self.evaluation_results['max_score']:.2f}",
            f"Min Score: {self.evaluation_results['min_score']:.2f}",
            f"Success Rate: {self.evaluation_results['success_rate']:.2%}",
            f"Mean Length: {self.evaluation_results['mean_length']:.2f} ± {self.evaluation_results['std_length']:.2f}"
        ]
        
        axes[1, 1].text(0.1, 0.5, '\n'.join(metrics), transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to a JSON file."""
        if not self.evaluation_results:
            print("No evaluation results to save.")
            return
            
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        print(f"Evaluation results saved to {filepath}")


def compare_agents(agents: Dict[str, Any], env, n_eval_episodes: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple agents on the same environment.
    
    Args:
        agents: Dictionary mapping agent names to agent objects
        env: Environment to evaluate on
        n_eval_episodes: Number of episodes to evaluate each agent
        
    Returns:
        Dictionary containing evaluation results for each agent
    """
    results = {}
    
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        evaluator = AgentEvaluator(agent, env, n_eval_episodes)
        results[name] = evaluator.evaluate()
    
    return results


def plot_comparison(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
    """
    Plot comparison results for multiple agents.
    
    Args:
        results: Results from compare_agents()
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    agent_names = list(results.keys())
    mean_scores = [results[name]['mean_score'] for name in agent_names]
    std_scores = [results[name]['std_score'] for name in agent_names]
    success_rates = [results[name]['success_rate'] for name in agent_names]
    
    # Mean scores comparison
    bars = axes[0, 0].bar(agent_names, mean_scores, yerr=std_scores, 
                         capsize=5, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_ylabel('Mean Score')
    axes[0, 0].set_title('Mean Score Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, mean_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Success rates comparison
    bars = axes[0, 1].bar(agent_names, success_rates, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Success Rate Comparison')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Score distributions
    for i, (name, result) in enumerate(results.items()):
        axes[1, 0].hist(result['scores'], bins=20, alpha=0.6, label=name, 
                       color=['skyblue', 'lightcoral', 'lightgreen'][i % 3])
    axes[1, 0].set_xlabel('Episode Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance summary table
    summary_data = []
    for name, result in results.items():
        summary_data.append([
            name,
            f"{result['mean_score']:.2f} ± {result['std_score']:.2f}",
            f"{result['max_score']:.2f}",
            f"{result['success_rate']:.1%}"
        ])
    
    table = axes[1, 1].table(cellText=summary_data,
                           colLabels=['Agent', 'Mean ± Std', 'Max Score', 'Success Rate'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Performance Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def compute_sample_efficiency(training_scores: Dict[str, List[float]], 
                            threshold: float = 195.0,
                            window_size: int = 100) -> Dict[str, int]:
    """
    Compute sample efficiency metrics for different agents.
    
    Args:
        training_scores: Dictionary mapping agent names to their training scores
        threshold: Performance threshold to reach
        window_size: Window size for moving average
        
    Returns:
        Dictionary mapping agent names to episodes needed to reach threshold
    """
    efficiency_results = {}
    
    for name, scores in training_scores.items():
        if len(scores) < window_size:
            efficiency_results[name] = None
            continue
            
        # Compute moving average
        moving_avg = []
        for i in range(window_size - 1, len(scores)):
            avg = np.mean(scores[i - window_size + 1:i + 1])
            moving_avg.append(avg)
        
        # Find first episode where threshold is reached
        episodes_to_threshold = None
        for i, avg in enumerate(moving_avg):
            if avg >= threshold:
                episodes_to_threshold = i + window_size
                break
        
        efficiency_results[name] = episodes_to_threshold
    
    return efficiency_results
