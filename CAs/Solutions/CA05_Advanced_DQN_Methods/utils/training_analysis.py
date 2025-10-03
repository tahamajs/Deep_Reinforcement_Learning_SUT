"""
DQN Training Analysis and Visualization
======================================

This module provides comprehensive analysis tools for DQN training,
including performance metrics, learning dynamics, and visualization utilities.

Author: CA5 Implementation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple
import warnings

warnings.filterwarnings("ignore")


class DQNAnalysis:
    """Analyze and visualize DQN training"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def plot_training_progress(self, scores, losses):
        """Plot comprehensive training analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        episodes = range(len(scores))
        
        # Plot 1: Episode rewards
        axes[0,0].plot(episodes, scores, alpha=0.6, color='blue', linewidth=1, label='Episode Scores')
        
        window = min(50, len(scores)//10) if len(scores) > 10 else 5
        if len(scores) >= window:
            moving_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
            axes[0,0].plot(episodes, moving_avg, color='red', linewidth=2, 
                          label=f'{window}-Episode Average')
        
        axes[0,0].set_title('Episode Rewards', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Training loss
        if hasattr(self.agent, 'losses') and len(self.agent.losses) > 0:
            loss_episodes = np.linspace(0, len(episodes), len(self.agent.losses))
            axes[0,1].plot(loss_episodes, self.agent.losses, alpha=0.6, color='orange')
            
            if len(self.agent.losses) > 100:
                window_loss = 100
                smoothed_loss = np.convolve(self.agent.losses, np.ones(window_loss)/window_loss, mode='valid')
                loss_smooth_episodes = np.linspace(0, len(episodes), len(smoothed_loss))
                axes[0,1].plot(loss_smooth_episodes, smoothed_loss, color='red', linewidth=2, label='Smoothed')
                axes[0,1].legend()
            
            axes[0,1].set_title('Training Loss', fontsize=12, fontweight='bold')
            axes[0,1].set_xlabel('Training Steps')
            axes[0,1].set_ylabel('MSE Loss')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Q-values
        if hasattr(self.agent, 'q_values') and len(self.agent.q_values) > 0:
            q_episodes = np.linspace(0, len(episodes), len(self.agent.q_values))
            axes[0,2].plot(q_episodes, self.agent.q_values, alpha=0.6, color='green')
            
            if len(self.agent.q_values) > 100:
                window_q = 100
                smoothed_q = np.convolve(self.agent.q_values, np.ones(window_q)/window_q, mode='valid')
                q_smooth_episodes = np.linspace(0, len(episodes), len(smoothed_q))
                axes[0,2].plot(q_smooth_episodes, smoothed_q, color='red', linewidth=2, label='Smoothed')
                axes[0,2].legend()
            
            axes[0,2].set_title('Average Q-Values', fontsize=12, fontweight='bold')
            axes[0,2].set_xlabel('Training Steps')
            axes[0,2].set_ylabel('Q-Value')
            axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Epsilon decay
        if hasattr(self.agent, 'epsilon'):
            # Reconstruct epsilon history
            epsilon_history = []
            epsilon = 1.0
            epsilon_decay = self.agent.epsilon_decay if hasattr(self.agent, 'epsilon_decay') else 0.995
            epsilon_min = self.agent.epsilon_min if hasattr(self.agent, 'epsilon_min') else 0.01
            
            for _ in range(len(episodes)):
                epsilon_history.append(epsilon)
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            axes[1,0].plot(episodes, epsilon_history, color='purple', linewidth=2)
            axes[1,0].set_title('Epsilon Decay (Exploration)', fontsize=12, fontweight='bold')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Epsilon')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Score distribution
        if len(scores) > 10:
            axes[1,1].hist(scores, bins=min(30, len(scores)//5), alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,1].axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.2f}')
            axes[1,1].axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.2f}')
            axes[1,1].set_title('Score Distribution', fontsize=12, fontweight='bold')
            axes[1,1].set_xlabel('Score')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Performance metrics
        axes[1,2].axis('off')
        
        final_100 = scores[-100:] if len(scores) >= 100 else scores
        final_50 = scores[-50:] if len(scores) >= 50 else scores
        
        metrics_text = f"""
Training Metrics Summary:
{'='*30}

Total Episodes: {len(scores)}
Mean Score: {np.mean(scores):.2f}
Std Score: {np.std(scores):.2f}
Max Score: {np.max(scores):.2f}
Min Score: {np.min(scores):.2f}

Last 100 Episodes:
  Mean: {np.mean(final_100):.2f}
  Std: {np.std(final_100):.2f}

Last 50 Episodes:
  Mean: {np.mean(final_50):.2f}
  Std: {np.std(final_50):.2f}

Buffer Size: {len(self.agent.memory) if hasattr(self.agent, 'memory') else 'N/A'}
Final Epsilon: {self.agent.epsilon:.4f} if hasattr(self.agent, 'epsilon') else 'N/A'}
"""
        
        axes[1,2].text(0.05, 0.95, metrics_text, transform=axes[1,2].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1,2].set_title('Training Metrics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_learning_dynamics(self, scores):
        """Analyze learning dynamics and convergence"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Cumulative reward
        cumulative_rewards = np.cumsum(scores)
        axes[0,0].plot(cumulative_rewards, color='blue', linewidth=2)
        axes[0,0].set_title('Cumulative Reward', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Cumulative Reward')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Episode length (if available)
        if hasattr(self.agent, 'episode_lengths'):
            axes[0,1].plot(self.agent.episode_lengths, alpha=0.6, color='green')
            axes[0,1].set_title('Episode Length', fontsize=12, fontweight='bold')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Steps')
            axes[0,1].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, 'Episode lengths not tracked', 
                          transform=axes[0,1].transAxes, ha='center', va='center')
            axes[0,1].set_title('Episode Length', fontsize=12, fontweight='bold')
        
        # 3. Learning rate (if available)
        if len(scores) > 10:
            window = min(50, len(scores)//10)
            learning_rates = []
            for i in range(window, len(scores)):
                recent_improvement = np.mean(scores[i-window:i]) - np.mean(scores[max(0,i-2*window):i-window])
                learning_rates.append(recent_improvement)
            
            axes[1,0].plot(range(window, len(scores)), learning_rates, color='orange', linewidth=2)
            axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1,0].set_title(f'Learning Progress (Window={window})', fontsize=12, fontweight='bold')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Score Improvement')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Variance over time
        if len(scores) > 20:
            window = min(20, len(scores)//10)
            variances = [np.var(scores[max(0, i-window):i+1]) for i in range(len(scores))]
            axes[1,1].plot(variances, color='purple', linewidth=2)
            axes[1,1].set_title(f'Score Variance (Window={window})', fontsize=12, fontweight='bold')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Variance')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_report(self, scores, losses):
        """Create a comprehensive text summary report"""
        print("\\n" + "="*60)
        print("DQN TRAINING SUMMARY REPORT")
        print("="*60)
        
        print(f"\\nTraining Configuration:")
        print(f"  Total Episodes: {len(scores)}")
        print(f"  State Size: {self.agent.state_size}")
        print(f"  Action Size: {self.agent.action_size}")
        print(f"  Learning Rate: {self.agent.lr}")
        print(f"  Gamma: {self.agent.gamma}")
        print(f"  Batch Size: {self.agent.batch_size}")
        
        print(f"\\nPerformance Metrics:")
        print(f"  Mean Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(f"  Median Score: {np.median(scores):.2f}")
        print(f"  Max Score: {np.max(scores):.2f}")
        print(f"  Min Score: {np.min(scores):.2f}")
        
        if len(scores) >= 100:
            final_100 = scores[-100:]
            print(f"\\nFinal 100 Episodes:")
            print(f"  Mean: {np.mean(final_100):.2f} ± {np.std(final_100):.2f}")
            print(f"  Max: {np.max(final_100):.2f}")
            print(f"  Min: {np.min(final_100):.2f}")
        
        if len(scores) >= 50:
            final_50 = scores[-50:]
            print(f"\\nFinal 50 Episodes:")
            print(f"  Mean: {np.mean(final_50):.2f} ± {np.std(final_50):.2f}")
            print(f"  Max: {np.max(final_50):.2f}")
            print(f"  Min: {np.min(final_50):.2f}")
        
        print(f"\\nTraining Dynamics:")
        print(f"  Total Reward: {np.sum(scores):.2f}")
        print(f"  Final Epsilon: {self.agent.epsilon:.4f}")
        if hasattr(self.agent, 'memory'):
            print(f"  Buffer Size: {len(self.agent.memory)}")
        
        if hasattr(self.agent, 'losses') and len(self.agent.losses) > 0:
            print(f"\\nLoss Statistics:")
            print(f"  Mean Loss: {np.mean(self.agent.losses):.4f}")
            print(f"  Final Loss: {np.mean(self.agent.losses[-100:]):.4f}")
        
        print("\\n" + "="*60)


def create_dqn_analysis(agent, scores, losses):
    """Create and return a DQN analysis instance with plots"""
    analyzer = DQNAnalysis(agent)
    analyzer.plot_training_progress(scores, losses)
    analyzer.analyze_learning_dynamics(scores)
    analyzer.create_summary_report(scores, losses)
    return analyzer


if __name__ == "__main__":
    print("DQN Training Analysis Module")
    print("This module provides comprehensive analysis tools for DQN training")
