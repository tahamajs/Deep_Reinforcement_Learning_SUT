"""
CA4: Policy Gradient Methods - Extended Training Examples
========================================================

This module provides comprehensive training examples and analysis functions
for policy gradient methods, including hyperparameter sensitivity studies,
curriculum learning demonstrations, and performance comparisons.

Author: DRL Course Team
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PolicyNetwork(nn.Module):
    """Simple policy network for policy gradient methods"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueNetwork(nn.Module):
    """Value network for baseline and actor-critic methods"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class REINFORCEAgent:
    """REINFORCE agent with optional baseline"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 use_baseline: bool = False,
                 entropy_coef: float = 0.0):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.use_baseline = use_baseline
        if use_baseline:
            self.value_net = ValueNetwork(state_dim)
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def update(self, trajectory: List[Tuple[np.ndarray, int, float, float, bool]]):
        """Update policy using REINFORCE"""
        states, actions, rewards, log_probs, dones = zip(*trajectory)

        # Calculate discounted returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.stack(policy_loss).sum()

        # Add entropy bonus
        if self.entropy_coef > 0:
            states_tensor = torch.FloatTensor(states)
            probs = self.policy(states_tensor)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            policy_loss -= self.entropy_coef * entropy

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Update baseline if used
        if self.use_baseline:
            states_tensor = torch.FloatTensor(states)
            values = self.value_net(states_tensor).squeeze()

            value_loss = nn.MSELoss()(values, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        return policy_loss.item()


def hyperparameter_sensitivity_study() -> pd.DataFrame:
    """
    Comprehensive hyperparameter sensitivity analysis for policy gradients

    Returns:
        DataFrame containing results of hyperparameter combinations
    """
    logger.info("Starting Policy Gradient Hyperparameter Sensitivity Analysis")

    # Define hyperparameter ranges to test
    learning_rates = [0.0001, 0.001, 0.01]
    gamma_values = [0.9, 0.95, 0.99, 0.995]
    baseline_options = [True, False]
    entropy_coeffs = [0.0, 0.01, 0.1]

    results = []

    logger.info(f"Testing {len(learning_rates) * len(gamma_values) * len(baseline_options) * len(entropy_coeffs)} hyperparameter combinations")

    for lr in learning_rates:
        for gamma in gamma_values:
            for use_baseline in baseline_options:
                for entropy_coef in entropy_coeffs:
                    # In practice, this would run actual training episodes
                    # Here we simulate results for demonstration
                    performance = {
                        'learning_rate': lr,
                        'gamma': gamma,
                        'use_baseline': use_baseline,
                        'entropy_coef': entropy_coef,
                        'final_reward': np.random.normal(100 + lr*1000 + gamma*50, 20),
                        'convergence_speed': np.random.exponential(2) * (1/lr),
                        'stability_score': np.random.beta(2, 2)
                    }
                    results.append(performance)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Learning rate impact
    lr_groups = df.groupby('learning_rate')['final_reward'].agg(['mean', 'std'])
    axes[0,0].errorbar(lr_groups.index, lr_groups['mean'], yerr=lr_groups['std'],
                      marker='o', linewidth=2, capsize=5)
    axes[0,0].set_xlabel('Learning Rate')
    axes[0,0].set_ylabel('Final Reward')
    axes[0,0].set_title('Learning Rate Impact')
    axes[0,0].set_xscale('log')
    axes[0,0].grid(True, alpha=0.3)

    # Gamma impact
    gamma_groups = df.groupby('gamma')['final_reward'].agg(['mean', 'std'])
    axes[0,1].errorbar(gamma_groups.index, gamma_groups['mean'], yerr=gamma_groups['std'],
                      marker='s', linewidth=2, capsize=5)
    axes[0,1].set_xlabel('Discount Factor (γ)')
    axes[0,1].set_ylabel('Final Reward')
    axes[0,1].set_title('Discount Factor Impact')
    axes[0,1].grid(True, alpha=0.3)

    # Baseline impact
    baseline_groups = df.groupby('use_baseline')['final_reward'].agg(['mean', 'std'])
    axes[0,2].bar(['No Baseline', 'With Baseline'],
                  baseline_groups['mean'], yerr=baseline_groups['std'],
                  capsize=5, alpha=0.7)
    axes[0,2].set_ylabel('Final Reward')
    axes[0,2].set_title('Baseline Impact')
    axes[0,2].grid(True, alpha=0.3)

    # Entropy coefficient impact
    entropy_groups = df.groupby('entropy_coef')['final_reward'].agg(['mean', 'std'])
    axes[1,0].errorbar(entropy_groups.index, entropy_groups['mean'], yerr=entropy_groups['std'],
                      marker='^', linewidth=2, capsize=5)
    axes[1,0].set_xlabel('Entropy Coefficient')
    axes[1,0].set_ylabel('Final Reward')
    axes[1,0].set_title('Entropy Regularization Impact')
    axes[1,0].set_xscale('log')
    axes[1,0].grid(True, alpha=0.3)

    # Convergence speed analysis
    axes[1,1].scatter(df['learning_rate'], df['convergence_speed'],
                     alpha=0.6, s=50, c=df['final_reward'], cmap='viridis')
    axes[1,1].set_xlabel('Learning Rate')
    axes[1,1].set_ylabel('Convergence Speed')
    axes[1,1].set_title('Learning Rate vs Convergence Speed')
    axes[1,1].set_xscale('log')
    axes[1,1].colorbar(label='Final Reward')

    # Stability analysis
    stability_pivot = df.pivot_table(values='stability_score',
                                   index='learning_rate',
                                   columns='gamma',
                                   aggfunc='mean')
    sns.heatmap(stability_pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=axes[1,2], cbar_kws={'label': 'Stability Score'})
    axes[1,2].set_title('Stability Matrix (LR × γ)')

    plt.tight_layout()
    plt.savefig('hyperparameter_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print insights
    logger.info("Hyperparameter sensitivity analysis completed")
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SENSITIVITY INSIGHTS")
    print("=" * 60)

    best_config = df.loc[df['final_reward'].idxmax()]
    print(f"Best performing configuration:")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Gamma: {best_config['gamma']}")
    print(f"  Use Baseline: {best_config['use_baseline']}")
    print(f"  Entropy Coef: {best_config['entropy_coef']}")
    print(".2f")

    # print("
# Key Insights:")
    print(f"• Learning rate impact: {lr_groups['mean'].max() - lr_groups['mean'].min():.1f} reward difference")
    print(f"• Gamma impact: {gamma_groups['mean'].max() - gamma_groups['mean'].min():.1f} reward difference")
    print(f"• Baseline benefit: {baseline_groups['mean'][True] - baseline_groups['mean'][False]:.1f} average improvement")
    # print("• Entropy regularization helps exploration but may slow convergence"
    return df


def curriculum_learning_example() -> Dict[str, Any]:
    """
    Demonstrate curriculum learning for policy gradients

    Returns:
        Dictionary containing curriculum learning results
    """
    logger.info("Starting Curriculum Learning Demonstration")

    # Define curriculum stages
    curriculum_stages = [
        {
            'name': 'Simple Tasks',
            'environments': ['CartPole-v0', 'MountainCar-v0'],
            'description': 'Basic control tasks with clear reward signals'
        },
        {
            'name': 'Complex Control',
            'environments': ['CartPole-v1', 'Pendulum-v1'],
            'description': 'More challenging control with continuous actions'
        },
        {
            'name': 'Advanced Tasks',
            'environments': ['LunarLander-v2', 'BipedalWalker-v3'],
            'description': 'Complex environments requiring sophisticated policies'
        }
    ]

    # Simulate curriculum learning progress
    stages_completed = []
    skills_learned = []

    for stage in curriculum_stages:
        logger.info(f"Starting {stage['name']} stage")
        print(f"\n{stage['name']} Stage:")
        print(f"Description: {stage['description']}")
        print(f"Environments: {', '.join(stage['environments'])}")

        # Mock training progress (in practice, this would train on actual environments)
        base_performance = len(stages_completed) * 20 + np.random.normal(50, 10)
        stage_performance = []

        for env in stage['environments']:
            performance = base_performance + np.random.normal(0, 15)
            stage_performance.append(performance)
            # print(".1f"
        avg_performance = np.mean(stage_performance)
        stages_completed.append(stage)

        # Determine skills learned
        if avg_performance > 70:
            skills_learned.append(f"Mastery of {stage['name'].lower()}")
        elif avg_performance > 50:
            skills_learned.append(f"Competence in {stage['name'].lower()}")
        else:
            skills_learned.append(f"Basic understanding of {stage['name'].lower()}")

    # print("
# 🎓 Curriculum Learning Summary:"    print(f"Stages Completed: {len(stages_completed)}")
    print(f"Skills Acquired: {len(skills_learned)}")
    print("\nSkills Learned:")
    for i, skill in enumerate(skills_learned, 1):
        print(f"{i}. {skill}")

    # Plot curriculum progress
    plt.figure(figsize=(12, 6))
    stage_names = [s['name'] for s in stages_completed]
    performances = [50, 70, 85]  # Mock progressive improvement

    plt.plot(stage_names, performances, 'bo-', linewidth=3, markersize=10, alpha=0.8)
    plt.fill_between(range(len(stage_names)),
                     np.array(performances) - 10,
                     np.array(performances) + 10,
                     alpha=0.3, color='blue')

    plt.xlabel('Curriculum Stage')
    plt.ylabel('Average Performance')
    plt.title('Curriculum Learning Progress')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)

    for i, (stage, perf) in enumerate(zip(stage_names, performances)):
        plt.annotate('.0f', (i, perf), xytext=(0, 10),
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    plt.savefig('curriculum_learning_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("Curriculum learning demonstration completed")
    return {
        'stages_completed': stages_completed,
        'skills_learned': skills_learned,
        'final_performance': performances[-1]
    }


def performance_comparison_study() -> Dict[str, Any]:
    """
    Compare different policy gradient variants

    Returns:
        Dictionary containing comparison results and data
    """
    logger.info("Starting Policy Gradient Variants Performance Comparison")

    # Define algorithms to compare
    algorithms = {
        'REINFORCE': {
            'description': 'Basic Monte Carlo policy gradient',
            'strengths': ['Simple implementation', 'Unbiased gradient estimates'],
            'weaknesses': ['High variance', 'Slow convergence', 'Sample inefficient']
        },
        'REINFORCE with Baseline': {
            'description': 'REINFORCE with value function baseline',
            'strengths': ['Reduced variance', 'Better sample efficiency'],
            'weaknesses': ['Requires value function training', 'More complex']
        },
        'Actor-Critic': {
            'description': 'Policy gradient with critic for TD learning',
            'strengths': ['Lower variance', 'Online learning', 'Better convergence'],
            'weaknesses': ['Bootstrapping bias', 'Stability issues']
        },
        'A2C (Advantage Actor-Critic)': {
            'description': 'Actor-Critic with advantage function',
            'strengths': ['Bias-variance tradeoff', 'Stable training', 'Good performance'],
            'weaknesses': ['Hyperparameter sensitive', 'Complex implementation']
        },
        'PPO': {
            'description': 'Proximal Policy Optimization',
            'strengths': ['Stable training', 'Good sample efficiency', 'Robust'],
            'weaknesses': ['Computationally intensive', 'Complex clipping']
        }
    }

    # Mock performance data for comparison (in practice, this would be real training results)
    environments = ['CartPole-v1', 'LunarLander-v2', 'Pendulum-v1']
    performance_data = {}

    for env in environments:
        performance_data[env] = {}
        for alg in algorithms.keys():
            # Generate realistic performance based on algorithm characteristics
            base_score = {'CartPole-v1': 200, 'LunarLander-v2': 100, 'Pendulum-v1': -200}[env]
            alg_multipliers = {
                'REINFORCE': 0.7,
                'REINFORCE with Baseline': 0.85,
                'Actor-Critic': 0.9,
                'A2C (Advantage Actor-Critic)': 0.95,
                'PPO': 1.0
            }
            score = base_score * alg_multipliers[alg] + np.random.normal(0, base_score * 0.1)
            performance_data[env][alg] = max(score, -500)  # Floor for pendulum

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Bar chart comparison
    env_names = list(performance_data.keys())
    alg_names = list(algorithms.keys())

    x = np.arange(len(env_names))
    width = 0.15
    multiplier = 0

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (alg, color) in enumerate(zip(alg_names, colors)):
        scores = [performance_data[env][alg] for env in env_names]
        offset = width * multiplier
        bars = axes[0,0].bar(x + offset, scores, width, label=alg, color=color, alpha=0.8)
        axes[0,0].bar_label(bars, fmt='.0f', padding=3, fontsize=8)
        multiplier += 1

    axes[0,0].set_xlabel('Environment')
    axes[0,0].set_ylabel('Average Reward')
    axes[0,0].set_title('Algorithm Performance Comparison')
    axes[0,0].set_xticks(x + width * 2, env_names)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)

    # Radar chart for algorithm characteristics
    categories = ['Sample Efficiency', 'Stability', 'Ease of Implementation', 'Convergence Speed', 'Final Performance']

    # Mock characteristics scores (0-10 scale)
    characteristics = {
        'REINFORCE': [3, 4, 9, 4, 6],
        'REINFORCE with Baseline': [5, 6, 7, 5, 7],
        'Actor-Critic': [7, 5, 6, 7, 8],
        'A2C (Advantage Actor-Critic)': [8, 7, 5, 8, 9],
        'PPO': [9, 9, 4, 6, 10]
    }

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    for alg, scores in characteristics.items():
        scores += scores[:1]  # Close the loop
        axes[0,1].plot(angles, scores, 'o-', linewidth=2, label=alg, markersize=6)

    axes[0,1].set_xticks(angles[:-1])
    axes[0,1].set_xticklabels(categories, fontsize=9)
    axes[0,1].set_ylim(0, 10)
    axes[0,1].set_title('Algorithm Characteristics Radar')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)

    # Learning curves comparison
    episodes = np.arange(0, 1000, 50)
    for alg in alg_names[:3]:  # Show first 3 for clarity
        # Mock learning curve with different convergence rates
        convergence_rates = {'REINFORCE': 0.002, 'REINFORCE with Baseline': 0.003, 'Actor-Critic': 0.004}
        noise_levels = {'REINFORCE': 50, 'REINFORCE with Baseline': 30, 'Actor-Critic': 20}

        final_score = 150
        curve = final_score * (1 - np.exp(-convergence_rates[alg] * episodes))
        curve += np.random.normal(0, noise_levels[alg], len(episodes))

        axes[1,0].plot(episodes, curve, label=alg, linewidth=2, alpha=0.8)

    axes[1,0].set_xlabel('Training Episodes')
    axes[1,0].set_ylabel('Average Reward')
    axes[1,0].set_title('Learning Curves Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Computational complexity vs performance
    complexities = [1, 2, 3, 4, 5]  # Relative complexity scores
    performances = [6, 7, 8, 9, 10]  # Relative performance scores

    axes[1,1].scatter(complexities, performances, s=100, alpha=0.7, c='red')
    for i, alg in enumerate(alg_names):
        axes[1,1].annotate(alg, (complexities[i], performances[i]),
                          xytext=(5, 5), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    axes[1,1].set_xlabel('Implementation Complexity')
    axes[1,1].set_ylabel('Performance Score')
    axes[1,1].set_title('Complexity vs Performance Tradeoff')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('policy_gradient_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed comparison
    logger.info("Performance comparison analysis completed")
    print("\n" + "=" * 55)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 55)

    for alg, info in algorithms.items():
        print(f"\n{alg}:")
        print(f"  Description: {info['description']}")
        print(f"  Strengths: {', '.join(info['strengths'])}")
        print(f"  Weaknesses: {', '.join(info['weaknesses'])}")

        # Calculate average performance across environments
        avg_perf = np.mean([performance_data[env][alg] for env in environments])
    #     print(".1f"
    # print("
# 💡 Recommendations:"    print("• For simple problems: Start with REINFORCE + Baseline")
    print("• For complex environments: Use A2C or PPO")
    print("• For research/prototyping: Actor-Critic variants")
    print("• For production systems: PPO (stable and robust)")

    return {
        'algorithms': algorithms,
        'performance_data': performance_data,
        'characteristics': characteristics
    }


def train_with_monitoring(env_name: str = 'CartPole-v1',
                         num_episodes: int = 1000,
                         lr: float = 0.001,
                         gamma: float = 0.99,
                         use_baseline: bool = True,
                         entropy_coef: float = 0.01) -> Dict[str, List[float]]:
    """
    Train REINFORCE agent with comprehensive monitoring

    Args:
        env_name: Gymnasium environment name
        num_episodes: Number of training episodes
        lr: Learning rate
        gamma: Discount factor
        use_baseline: Whether to use value baseline
        entropy_coef: Entropy regularization coefficient

    Returns:
        Dictionary containing training metrics
    """
    logger.info(f"Starting training on {env_name} with monitoring")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, lr, gamma, use_baseline, entropy_coef)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    policy_losses = []
    value_losses = [] if use_baseline else None
    entropies = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        trajectory = []

        done = False
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory.append((state, action, reward, log_prob, done))
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Update agent
        loss = agent.update(trajectory)
        policy_losses.append(loss)

        # Calculate trajectory entropy
        states, actions, rewards, log_probs, dones = zip(*trajectory)
        entropy = -np.mean(log_probs)  # Approximate entropy
        entropies.append(entropy)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}")

    env.close()

    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'policy_losses': policy_losses,
        'entropies': entropies
    }

    if use_baseline:
        results['value_losses'] = value_losses

    logger.info("Training completed")
    return results


def plot_training_analysis(results: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Create comprehensive training analysis plots

    Args:
        results: Training results dictionary
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    episodes = range(1, len(results['episode_rewards']) + 1)

    # Episode rewards
    axes[0,0].plot(episodes, results['episode_rewards'], alpha=0.7)
    axes[0,0].plot(episodes, pd.Series(results['episode_rewards']).rolling(50).mean(),
                   linewidth=2, color='red', label='Rolling Mean (50)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Episode Reward')
    axes[0,0].set_title('Training Rewards')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0,1].plot(episodes, results['episode_lengths'], alpha=0.7)
    axes[0,1].plot(episodes, pd.Series(results['episode_lengths']).rolling(50).mean(),
                   linewidth=2, color='red', label='Rolling Mean (50)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Episode Length')
    axes[0,1].set_title('Episode Lengths')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Policy losses
    axes[0,2].plot(episodes, results['policy_losses'], alpha=0.7)
    axes[0,2].plot(episodes, pd.Series(results['policy_losses']).rolling(50).mean(),
                   linewidth=2, color='red', label='Rolling Mean (50)')
    axes[0,2].set_xlabel('Episode')
    axes[0,2].set_ylabel('Policy Loss')
    axes[0,2].set_title('Policy Loss')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)

    # Entropy evolution
    axes[1,0].plot(episodes, results['entropies'], alpha=0.7)
    axes[1,0].plot(episodes, pd.Series(results['entropies']).rolling(50).mean(),
                   linewidth=2, color='red', label='Rolling Mean (50)')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Policy Entropy')
    axes[1,0].set_title('Policy Entropy Evolution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Reward distribution (last 100 episodes)
    if len(results['episode_rewards']) >= 100:
        recent_rewards = results['episode_rewards'][-100:]
        axes[1,1].hist(recent_rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].axvline(np.mean(recent_rewards), color='red', linestyle='--',
                         label=f'Mean: {np.mean(recent_rewards):.1f}')
        axes[1,1].set_xlabel('Episode Reward')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Reward Distribution (Last 100 Episodes)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

    # Learning stability (reward variance)
    reward_variance = pd.Series(results['episode_rewards']).rolling(50).var()
    axes[1,2].plot(episodes, reward_variance, alpha=0.7)
    axes[1,2].set_xlabel('Episode')
    axes[1,2].set_ylabel('Reward Variance')
    axes[1,2].set_title('Learning Stability (Rolling Variance)')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("CA4: Policy Gradient Methods - Extended Training Examples")
    print("=" * 60)

    # Run hyperparameter sensitivity study
    print("\n1. Running Hyperparameter Sensitivity Study...")
    hp_results = hyperparameter_sensitivity_study()

    # Run curriculum learning example
    print("\n2. Running Curriculum Learning Example...")
    curriculum_results = curriculum_learning_example()

    # Run performance comparison
    print("\n3. Running Performance Comparison Study...")
    comparison_results = performance_comparison_study()

    # Example training with monitoring
    print("\n4. Running Example Training with Monitoring...")
    training_results = train_with_monitoring(
        env_name='CartPole-v1',
        num_episodes=500,
        lr=0.001,
        gamma=0.99,
        use_baseline=True,
        entropy_coef=0.01
    )

    # Plot training analysis
    print("\n5. Creating Training Analysis Plots...")
    plot_training_analysis(training_results, save_path='training_analysis.png')

    print("\n✅ All extended training examples completed!")
    print("Generated files:")
    print("- hyperparameter_sensitivity_analysis.png")
    print("- curriculum_learning_progress.png")
    print("- policy_gradient_comparison.png")
    print("- training_analysis.png")