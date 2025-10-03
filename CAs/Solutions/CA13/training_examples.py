import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import time
from typing import Dict, List, Any, Optional


def train_dqn_agent(env, agent, num_episodes=200, max_steps=500, eval_interval=20, 
                   target_reward=195, verbose=True):
    """
    Train a DQN agent and return training statistics
    
    Args:
        env: Environment
        agent: DQN agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_interval: Episodes between evaluations
        target_reward: Target reward for success
        verbose: Print progress
    
    Returns:
        Dictionary with training results
    """
    episode_rewards = []
    episode_lengths = []
    losses = []
    eval_rewards = []
    eval_lengths = []
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select action
            action = agent.act(obs)
            
            # Take step
            next_obs, reward, done, _ = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Update agent
            if len(agent.replay_buffer) > 32:
                loss = agent.update(batch_size=32)
            if loss is not None:
                    episode_losses.append(loss)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        if episode_losses:
            losses.extend(episode_losses)
        
        # Evaluation
        if episode % eval_interval == 0:
            eval_reward, eval_length = evaluate_agent_episode(env, agent, n_episodes=5)
            eval_rewards.append(eval_reward)
            eval_lengths.append(eval_length)
            
            if eval_reward > best_reward:
                best_reward = eval_reward
            
            if verbose:
                avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
                print(f"Episode {episode:4d} | Avg Reward: {avg_reward:6.2f} | "
                      f"Eval Reward: {eval_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")
        
        # Early stopping if target reached
        if len(eval_rewards) > 0 and eval_rewards[-1] >= target_reward:
            if verbose:
                print(f"Target reward {target_reward} reached at episode {episode}")
            break
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'length': episode_lengths
    })
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'losses': losses,
        'eval_rewards': eval_rewards,
        'eval_lengths': eval_lengths,
        'best_reward': best_reward,
        'episode_dataframe': results_df,
        'training_steps': len(losses)
    }


def train_model_based_agent(env, agent, num_episodes=200, max_steps=500, eval_interval=20,
                           planning_steps=10, target_reward=195, verbose=True):
    """
    Train a model-based agent with planning
    
    Args:
        env: Environment
        agent: Model-based agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_interval: Episodes between evaluations
        planning_steps: Number of planning steps per real step
        target_reward: Target reward for success
        verbose: Print progress
    
    Returns:
        Dictionary with training results
    """
    episode_rewards = []
    episode_lengths = []
    model_losses = []
    q_losses = []
    eval_rewards = []
    eval_lengths = []
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_model_losses = []
        episode_q_losses = []
        
        for step in range(max_steps):
            # Select action (with planning)
            if episode > 10:  # Start planning after some exploration
                action = agent.plan(obs)
            else:
                action = agent.act(obs, epsilon=0.5)
            
            # Take step
            next_obs, reward, done, _ = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Update agent
            if len(agent.replay_buffer) > 32:
                losses = agent.update(batch_size=32)
                if losses:
                    episode_model_losses.append(losses.get('dynamics_loss', 0) + losses.get('reward_loss', 0))
                    episode_q_losses.append(losses.get('q_loss', 0))
            
            # Planning steps
            for _ in range(planning_steps):
                if len(agent.replay_buffer) > 32:
                    agent.update(batch_size=32)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        if episode_model_losses:
            model_losses.extend(episode_model_losses)
        if episode_q_losses:
            q_losses.extend(episode_q_losses)
        
        # Evaluation
        if episode % eval_interval == 0:
            eval_reward, eval_length = evaluate_agent_episode(env, agent, n_episodes=5)
            eval_rewards.append(eval_reward)
            eval_lengths.append(eval_length)
            
            if eval_reward > best_reward:
                best_reward = eval_reward
            
            if verbose:
                avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
                avg_model_loss = np.mean(episode_model_losses) if episode_model_losses else 0
                avg_q_loss = np.mean(episode_q_losses) if episode_q_losses else 0
                print(f"Episode {episode:4d} | Avg Reward: {avg_reward:6.2f} | "
                      f"Eval Reward: {eval_reward:6.2f} | Model Loss: {avg_model_loss:.4f} | Q Loss: {avg_q_loss:.4f}")
        
        # Early stopping if target reached
        if len(eval_rewards) > 0 and eval_rewards[-1] >= target_reward:
            if verbose:
                print(f"Target reward {target_reward} reached at episode {episode}")
            break
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'length': episode_lengths
    })
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'model_losses': model_losses,
        'q_losses': q_losses,
        'eval_rewards': eval_rewards,
        'eval_lengths': eval_lengths,
        'best_reward': best_reward,
        'episode_dataframe': results_df,
        'training_steps': len(model_losses)
    }


def evaluate_agent(env, agent, num_episodes=10, max_steps=500, render=False):
    """
    Evaluate an agent for multiple episodes
    
    Args:
        env: Environment
        agent: Agent to evaluate
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render episodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # Select action (no exploration)
            if hasattr(agent, 'act'):
                action = agent.act(obs, epsilon=0.0)
            else:
                action = env.action_space.sample()
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        if render:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {step + 1}")
    
    return {
        'mean_return': np.mean(episode_rewards),
        'std_return': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def evaluate_agent_episode(env, agent, n_episodes=5, max_steps=500):
    """
    Quick evaluation for a single agent
    
    Returns:
        (mean_reward, mean_length)
    """
    rewards = []
    lengths = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.act(obs, epsilon=0.0)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        lengths.append(step + 1)
    
    return np.mean(rewards), np.mean(lengths)


def compare_agents(env, agents, num_episodes=100, eval_episodes=10):
    """
    Compare multiple agents on the same environment
    
    Args:
        env: Environment
        agents: Dictionary of agent_name -> agent
        num_episodes: Training episodes
        eval_episodes: Evaluation episodes
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\nTraining {agent_name}...")
        
        # Train agent
        if hasattr(agent, 'replay_buffer'):  # DQN-style agent
            train_results = train_dqn_agent(env, agent, num_episodes, verbose=False)
        elif hasattr(agent, 'store_experience'):  # Model-based agent
            train_results = train_model_based_agent(env, agent, num_episodes, verbose=False)
        else:
            print(f"Unknown agent type for {agent_name}, skipping...")
            continue
        
        # Evaluate agent
        eval_results = evaluate_agent(env, agent, eval_episodes)
        
        results[agent_name] = {
            'training': train_results,
            'evaluation': eval_results
        }
        
        print(f"{agent_name}: Final reward = {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")

    return results


def plot_training_curves(results, save_path=None):
    """
    Plot training curves for multiple agents
    
    Args:
        results: Dictionary from compare_agents
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training rewards
    ax = axes[0, 0]
    for agent_name, result in results.items():
        rewards = result['training']['rewards']
        window = max(1, len(rewards) // 50)
        smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, label=agent_name, linewidth=2)
    
    ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Evaluation rewards
    ax = axes[0, 1]
    for agent_name, result in results.items():
        eval_rewards = result['training'].get('eval_rewards', [])
        if eval_rewards:
            ax.plot(eval_rewards, label=agent_name, marker='o', linewidth=2)
    
    ax.set_title('Evaluation Rewards', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Epoch')
    ax.set_ylabel('Average Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Episode lengths
    ax = axes[1, 0]
    for agent_name, result in results.items():
        lengths = result['training']['lengths']
        window = max(1, len(lengths) // 50)
        smoothed = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
        ax.plot(smoothed, label=agent_name, linewidth=2)
    
    ax.set_title('Episode Lengths', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Length')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Final performance comparison
    ax = axes[1, 1]
    agent_names = list(results.keys())
    final_rewards = [results[name]['evaluation']['mean_return'] for name in agent_names]
    final_stds = [results[name]['evaluation']['std_return'] for name in agent_names]
    
    bars = ax.bar(agent_names, final_rewards, yerr=final_stds, capsize=5, alpha=0.7)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def hyperparameter_sweep(env, agent_class, param_grid, num_episodes=50, eval_episodes=5):
    """
    Perform hyperparameter sweep for an agent
    
    Args:
        env: Environment
        agent_class: Agent class to instantiate
        param_grid: Dictionary of parameter_name -> list of values
        num_episodes: Training episodes per configuration
        eval_episodes: Evaluation episodes per configuration
    
    Returns:
        DataFrame with results for all parameter combinations
    """
    import itertools
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    
    for i, combination in enumerate(combinations):
        params = dict(zip(param_names, combination))
        
        print(f"\nConfiguration {i+1}/{len(combinations)}: {params}")
        
        # Create agent with these parameters
        agent = agent_class(**params)
        
        # Train agent
        train_results = train_dqn_agent(env, agent, num_episodes, verbose=False)
        
        # Evaluate agent
        eval_results = evaluate_agent(env, agent, eval_episodes)
        
        # Store results
        result = params.copy()
        result.update({
            'final_reward': eval_results['mean_return'],
            'final_std': eval_results['std_return'],
            'training_reward': np.mean(train_results['rewards'][-10:]) if len(train_results['rewards']) >= 10 else np.mean(train_results['rewards']),
            'training_steps': train_results['training_steps']
        })
        
        results.append(result)
        
        print(f"Final reward: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")
    
    return pd.DataFrame(results)


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get appropriate device (CUDA if available)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")