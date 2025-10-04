import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
import os


def evaluate_agent(agent, env, num_episodes=100, save_results=True, save_dir="visualizations"):
    """
    Comprehensive evaluation of an RL agent
    
    Args:
        agent: Trained RL agent
        env: Environment to evaluate on
        num_episodes: Number of episodes for evaluation
        save_results: Whether to save results to file
        save_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating agent for {num_episodes} episodes...")
    
    rewards = []
    steps = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while episode_steps < 200:  # Max steps per episode
            if hasattr(agent, 'get_action'):
                action = agent.get_action(state, explore=False)
            else:
                action = agent.get_greedy_action(state)
                
            if action is None:
                break
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                if reward > 5:  # Successful episode (reached goal)
                    success_count += 1
                break
        
        rewards.append(episode_reward)
        steps.append(episode_steps)
    
    # Calculate metrics
    metrics = {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'avg_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'success_rate': success_count / num_episodes,
        'episodes': num_episodes,
        'rewards': rewards,
        'steps': steps
    }
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Average Steps: {metrics['avg_steps']:.1f} ± {metrics['std_steps']:.1f}")
    
    if save_results:
        save_evaluation_results(metrics, agent.__class__.__name__, save_dir)
    
    return metrics


def compare_agents(agents_dict, env, num_episodes=100, save_dir="visualizations"):
    """
    Compare multiple agents and generate comparison plots
    
    Args:
        agents_dict: Dictionary of {name: agent} pairs
        env: Environment for evaluation
        num_episodes: Number of episodes per agent
        save_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    print("Comparing multiple agents...")
    
    comparison_results = {}
    
    for name, agent in agents_dict.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_agent(agent, env, num_episodes, save_results=False)
        comparison_results[name] = metrics
    
    # Generate comparison plots
    create_comparison_plots(comparison_results, save_dir)
    
    # Save comparison results
    save_comparison_results(comparison_results, save_dir)
    
    return comparison_results


def analyze_performance(agent, save_dir="visualizations"):
    """
    Analyze agent's training performance and generate plots
    
    Args:
        agent: Trained agent with episode_rewards attribute
        save_dir: Directory to save plots
        
    Returns:
        Dictionary with performance analysis
    """
    print("Analyzing agent performance...")
    
    if not hasattr(agent, 'episode_rewards') or not agent.episode_rewards:
        print("No training data available for analysis")
        return {}
    
    rewards = agent.episode_rewards
    
    # Calculate performance metrics
    analysis = {
        'total_episodes': len(rewards),
        'final_performance': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
        'best_performance': np.max(rewards),
        'convergence_episode': find_convergence_episode(rewards),
        'learning_rate': calculate_learning_rate(rewards),
        'stability': calculate_stability(rewards)
    }
    
    # Generate performance plots
    create_performance_plots(agent, analysis, save_dir)
    
    # Save analysis
    save_performance_analysis(analysis, agent.__class__.__name__, save_dir)
    
    return analysis


def find_convergence_episode(rewards, window=50, threshold=0.1):
    """Find episode where performance converges"""
    if len(rewards) < window:
        return len(rewards)
    
    for i in range(window, len(rewards)):
        recent_avg = np.mean(rewards[i-window:i])
        earlier_avg = np.mean(rewards[i-window*2:i-window])
        
        if abs(recent_avg - earlier_avg) < threshold:
            return i
    
    return len(rewards)


def calculate_learning_rate(rewards, window=100):
    """Calculate learning rate (improvement per episode)"""
    if len(rewards) < window:
        return 0
    
    early_avg = np.mean(rewards[:window])
    late_avg = np.mean(rewards[-window:])
    
    return (late_avg - early_avg) / len(rewards)


def calculate_stability(rewards, window=50):
    """Calculate stability (low variance in recent episodes)"""
    if len(rewards) < window:
        return np.std(rewards)
    
    recent_rewards = rewards[-window:]
    return np.std(recent_rewards)


def create_comparison_plots(results, save_dir):
    """Create comparison plots for multiple agents"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward comparison
    ax1 = axes[0, 0]
    names = list(results.keys())
    avg_rewards = [results[name]['avg_reward'] for name in names]
    std_rewards = [results[name]['std_reward'] for name in names]
    
    bars = ax1.bar(names, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Success rate comparison
    ax2 = axes[0, 1]
    success_rates = [results[name]['success_rate'] * 100 for name in names]
    ax2.bar(names, success_rates, alpha=0.7, color='green')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Steps comparison
    ax3 = axes[1, 0]
    avg_steps = [results[name]['avg_steps'] for name in names]
    std_steps = [results[name]['std_steps'] for name in names]
    ax3.bar(names, avg_steps, yerr=std_steps, capsize=5, alpha=0.7, color='orange')
    ax3.set_ylabel('Average Steps')
    ax3.set_title('Steps to Goal Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # Reward distribution
    ax4 = axes[1, 1]
    for name in names:
        ax4.hist(results[name]['rewards'], alpha=0.5, label=name, bins=20)
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'agent_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_plots(agent, analysis, save_dir):
    """Create performance analysis plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning curve
    ax1 = axes[0, 0]
    ax1.plot(agent.episode_rewards, alpha=0.6, color='blue')
    
    # Add moving average
    window = 50
    if len(agent.episode_rewards) >= window:
        moving_avg = pd.Series(agent.episode_rewards).rolling(window=window).mean()
        ax1.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')
        ax1.legend()
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title(f'{agent.__class__.__name__} Learning Curve')
    ax1.grid(True, alpha=0.3)
    
    # Reward distribution
    ax2 = axes[0, 1]
    ax2.hist(agent.episode_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(analysis['final_performance'], color='red', linestyle='--', 
                label=f"Final: {analysis['final_performance']:.2f}")
    ax2.set_xlabel('Episode Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Performance metrics
    ax3 = axes[1, 0]
    metrics = ['Final Performance', 'Best Performance', 'Learning Rate', 'Stability']
    values = [
        analysis['final_performance'],
        analysis['best_performance'],
        analysis['learning_rate'] * 1000,  # Scale for visibility
        analysis['stability']
    ]
    bars = ax3.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    ax3.set_ylabel('Value')
    ax3.set_title('Performance Metrics')
    ax3.tick_params(axis='x', rotation=45)
    
    # Convergence analysis
    ax4 = axes[1, 1]
    if analysis['convergence_episode'] < len(agent.episode_rewards):
        ax4.axvline(analysis['convergence_episode'], color='red', linestyle='--', 
                   label=f"Convergence: Episode {analysis['convergence_episode']}")
    
    ax4.plot(agent.episode_rewards, alpha=0.6, color='blue')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Episode Reward')
    ax4.set_title('Convergence Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{agent.__class__.__name__}_performance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def save_evaluation_results(metrics, agent_name, save_dir):
    """Save evaluation results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Remove lists for JSON serialization
    save_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    save_metrics['agent_name'] = agent_name
    
    filename = os.path.join(save_dir, f'{agent_name}_evaluation_results.json')
    with open(filename, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    
    print(f"Evaluation results saved to {filename}")


def save_comparison_results(results, save_dir):
    """Save comparison results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Remove lists for JSON serialization
    save_results = {}
    for agent_name, metrics in results.items():
        save_results[agent_name] = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    
    filename = os.path.join(save_dir, 'agent_comparison_results.json')
    with open(filename, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"Comparison results saved to {filename}")


def save_performance_analysis(analysis, agent_name, save_dir):
    """Save performance analysis to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    
    analysis['agent_name'] = agent_name
    
    filename = os.path.join(save_dir, f'{agent_name}_performance_analysis.json')
    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Performance analysis saved to {filename}")
