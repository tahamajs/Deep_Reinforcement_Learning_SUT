import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional

def plot_learning_curve(episode_rewards, title="Learning Curve"):
    """Plot learning curve showing episode rewards over time"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, color='blue', linewidth=0.8)

    window_size = 50
    if len(episode_rewards) >= window_size:
        moving_avg = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(episode_rewards[start_idx:i+1]))
        plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size} episodes)')
        plt.legend()

    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title(f'{title} - Episode Rewards')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Learning Statistics:")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Reward std: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")

def plot_q_learning_analysis(agent):
    """Comprehensive analysis of Q-Learning performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    ax1.plot(agent.episode_rewards, alpha=0.6, color='blue', linewidth=0.8, label='Episode Reward')

    window = 50
    if len(agent.episode_rewards) >= window:
        moving_avg = pd.Series(agent.episode_rewards).rolling(window=window).mean()
        ax1.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Q-Learning: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(agent.episode_steps, alpha=0.7, color='green', linewidth=0.8)

    if len(agent.episode_steps) >= window:
        steps_avg = pd.Series(agent.episode_steps).rolling(window=window).mean()
        ax2.plot(steps_avg, color='darkgreen', linewidth=2, label=f'Moving Average ({window})')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Goal')
    ax2.set_title('Q-Learning: Steps per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(agent.epsilon_history, color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon (ε)')
    ax3.set_title('Exploration Rate Decay')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    final_rewards = agent.episode_rewards[-200:]  # Last 200 episodes
    ax4.hist(final_rewards, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(np.mean(final_rewards), color='red', linestyle='--',
                label=f'Mean: {np.mean(final_rewards):.2f}')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Final Performance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def show_q_values(agent, states_to_show=[(0,0), (1,0), (2,0), (0,1), (2,2)]):
    """Display Q-values for specific states"""
    print("\nLearned Q-values for key states:")
    print("State\t\tAction\t\tQ-value")
    print("-" * 40)

    for state in states_to_show:
        if not agent.env.is_terminal(state):
            valid_actions = agent.env.get_valid_actions(state)
            for action in valid_actions:
                q_val = agent.Q[state][action]
                print(f"{state}\t\t{action}\t\t{q_val:.3f}")
            print("-" * 40)

def compare_algorithms(td_agent, q_agent, sarsa_agent, V_td, V_optimal, V_sarsa, evaluation, sarsa_evaluation):
    """Compare TD(0), Q-Learning, and SARSA performance"""

    print("=" * 80)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 80)

    algorithms = {
        'TD(0)': {
            'agent': td_agent,
            'type': 'Policy Evaluation',
            'policy_type': 'Model-free evaluation',
            'learned_values': V_td,
            'evaluation': None
        },
        'Q-Learning': {
            'agent': q_agent,
            'type': 'Off-policy Control',
            'policy_type': 'Optimal policy',
            'learned_values': V_optimal,
            'evaluation': evaluation
        },
        'SARSA': {
            'agent': sarsa_agent,
            'type': 'On-policy Control',
            'policy_type': 'Behavior policy',
            'learned_values': V_sarsa,
            'evaluation': sarsa_evaluation
        }
    }

    print("\n1. PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"{'Algorithm':<12} {'Type':<20} {'Avg Reward':<12} {'Success Rate':<12}")
    print("-" * 50)

    for name, info in algorithms.items():
        if info['evaluation']:
            avg_reward = info['evaluation']['avg_reward']
            success_rate = info['evaluation']['success_rate'] * 100
            print(f"{name:<12} {info['type']:<20} {avg_reward:<12.2f} {success_rate:<12.1f}%")
        else:
            print(f"{name:<12} {info['type']:<20} {'N/A':<12} {'N/A':<12}")

    print("\n2. LEARNING CURVES COMPARISON")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    if hasattr(td_agent, 'episode_rewards'):
        plt.plot(td_agent.episode_rewards, label='TD(0)', alpha=0.7, color='blue')
    plt.plot(q_agent.episode_rewards, label='Q-Learning', alpha=0.7, color='red')
    plt.plot(sarsa_agent.episode_rewards, label='SARSA', alpha=0.7, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    window = 50

    if len(q_agent.episode_rewards) >= window:
        q_avg = pd.Series(q_agent.episode_rewards).rolling(window=window).mean()
        plt.plot(q_avg, label='Q-Learning', linewidth=2, color='red')

    if len(sarsa_agent.episode_rewards) >= window:
        sarsa_avg = pd.Series(sarsa_agent.episode_rewards).rolling(window=window).mean()
        plt.plot(sarsa_avg, label='SARSA', linewidth=2, color='green')

    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title(f'Moving Average ({window} episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(q_agent.epsilon_history, label='Q-Learning', color='red')
    plt.plot(sarsa_agent.epsilon_history, label='SARSA', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon (ε)')
    plt.title('Exploration Rate Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n3. VALUE FUNCTION COMPARISON")
    env = q_agent.env  # Get environment from agent
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if V_td:
        grid_td = np.zeros((env.size, env.size))
        for i, j in env.obstacles:
            grid_td[i, j] = min(V_td.values()) - 1
        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state not in env.obstacles:
                    grid_td[i, j] = V_td.get(state, 0)

        im1 = axes[0].imshow(grid_td, cmap='RdYlGn', aspect='equal')
        axes[0].set_title('TD(0) Values')
        plt.colorbar(im1, ax=axes[0])

    grid_q = np.zeros((env.size, env.size))
    for i, j in env.obstacles:
        grid_q[i, j] = min(V_optimal.values()) - 1
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state not in env.obstacles:
                grid_q[i, j] = V_optimal.get(state, 0)

    im2 = axes[1].imshow(grid_q, cmap='RdYlGn', aspect='equal')
    axes[1].set_title('Q-Learning Values')
    plt.colorbar(im2, ax=axes[1])

    grid_s = np.zeros((env.size, env.size))
    for i, j in env.obstacles:
        grid_s[i, j] = min(V_sarsa.values()) - 1
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state not in env.obstacles:
                grid_s[i, j] = V_sarsa.get(state, 0)

    im3 = axes[2].imshow(grid_s, cmap='RdYlGn', aspect='equal')
    axes[2].set_title('SARSA Values')
    plt.colorbar(im3, ax=axes[2])

    for ax in axes:
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))

    plt.tight_layout()
    plt.show()

    print("\n4. STATISTICAL ANALYSIS")
    print("-" * 50)

    key_states = [(0, 0), (1, 0), (2, 0), (3, 2), (2, 2)]
    print(f"{'State':<10} {'TD(0)':<10} {'Q-Learning':<12} {'SARSA':<10} {'Q-S Diff':<10}")
    print("-" * 55)

    for state in key_states:
        td_val = V_td.get(state, 0) if V_td else 0
        q_val = V_optimal.get(state, 0)
        s_val = V_sarsa.get(state, 0)
        diff = abs(q_val - s_val)

        print(f"{str(state):<10} {td_val:<10.2f} {q_val:<12.2f} {s_val:<10.2f} {diff:<10.3f}")

    return algorithms