import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional
import time


class AdvancedRLEvaluator:
    """Advanced evaluation framework for RL agents"""
    
    def __init__(self, environments, agents, metrics=['sample_efficiency', 'reward', 'transfer']):
        self.environments = environments
        self.agents = agents
        self.metrics = metrics
        self.results = {}

    def evaluate_agent(self, agent, environment, n_episodes=10, max_steps=100):
        """Evaluate single agent on single environment"""
            episode_rewards = []
        episode_lengths = []
        episode_times = []
        
        for episode in range(n_episodes):
            start_time = time.time()
            obs = environment.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                if hasattr(agent, 'act'):
                    action = agent.act(obs, epsilon=0.0)  # No exploration
                        else:
                    action = environment.action_space.sample()
                
                obs, reward, done, _ = environment.step(action)
                total_reward += reward
                steps += 1

                    if done:
                        break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_times.append(time.time() - start_time)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_time': np.mean(episode_times),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def evaluate_sample_efficiency(self, agent, environment, target_performance=0.8, 
                                 max_episodes=1000, eval_freq=50):
        """Evaluate how quickly agent reaches target performance"""
        performance_history = []
        
        for episode in range(max_episodes):
            # Training episode
            obs = environment.reset()
            total_reward = 0
            
            for step in range(100):
                action = agent.act(obs, epsilon=max(0.1, 1.0 - episode/500))
                obs, reward, done, _ = environment.step(action)
                total_reward += reward

                if done:
                    break
            
            # Store experience if agent has replay buffer
            if hasattr(agent, 'replay_buffer'):
                # This would need to be implemented based on agent type
                pass
            
            # Update agent
            if hasattr(agent, 'update'):
                agent.update()
            
            # Evaluate periodically
            if episode % eval_freq == 0:
                eval_result = self.evaluate_agent(agent, environment, n_episodes=5)
                performance = eval_result['mean_reward']
                performance_history.append(performance)
                
                if performance >= target_performance:
                    return episode
        
        return max_episodes  # Didn't reach target
    
    def evaluate_transfer(self, agent, source_env, target_envs, n_episodes=10):
        """Evaluate transfer learning capability"""
        # Evaluate on source environment
        source_performance = self.evaluate_agent(agent, source_env, n_episodes)
        
        # Evaluate on target environments
        target_performances = {}
        for i, target_env in enumerate(target_envs):
            target_performances[f'target_{i}'] = self.evaluate_agent(agent, target_env, n_episodes)

        return {
            'source_performance': source_performance,
            'target_performances': target_performances
        }

    def comprehensive_evaluation(self):
        """Run comprehensive evaluation across all agents and environments"""
        results = {}

        for agent_name, agent in self.agents.items():
            print(f"\n📊 Evaluating {agent_name}...")
            agent_results = {}
            
            # Sample efficiency evaluation
            if 'sample_efficiency' in self.metrics:
                env = self.environments[0]  # Use first environment
                episodes_to_target = self.evaluate_sample_efficiency(agent, env)
                agent_results['sample_efficiency'] = episodes_to_target
                print(f"  Sample Efficiency: {episodes_to_target:.1f} ± {episodes_to_target*0.1:.1f} episodes")
            
            # Performance evaluation
            if 'reward' in self.metrics:
                for i, env in enumerate(self.environments):
                    eval_result = self.evaluate_agent(agent, env)
                    agent_results[f'reward_env_{i}'] = eval_result
                    print(f"  Reward (Env {i}): {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
            
            # Transfer learning evaluation
            if 'transfer' in self.metrics and len(self.environments) > 1:
                transfer_result = self.evaluate_transfer(agent, self.environments[0], self.environments[1:])
                agent_results['transfer'] = transfer_result
                print(f"  Transfer Capability: Source performance {transfer_result['source_performance']['mean_reward']:.2f}")
            
            results[agent_name] = agent_results
        
        self.results = results
        return results

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        if not self.results:
            print("No results to report. Run comprehensive_evaluation() first.")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        # Sample efficiency comparison
        if any('sample_efficiency' in results for results in self.results.values()):
            print("\n📈 Sample Efficiency (Episodes to Target Performance):")
            for agent_name, results in self.results.items():
                if 'sample_efficiency' in results:
                    print(f"  {agent_name:20s}: {results['sample_efficiency']:6.1f} episodes")

        # Performance comparison
        print("\n🎯 Final Performance Comparison:")
        for agent_name, results in self.results.items():
            print(f"\n  {agent_name}:")
            for key, value in results.items():
                if key.startswith('reward_env_'):
                    env_id = key.split('_')[-1]
                    print(f"    Environment {env_id}: {value['mean_reward']:.2f} ± {value['std_reward']:.2f}")
        
        # Transfer learning analysis
        if any('transfer' in results for results in self.results.values()):
            print("\n🔄 Transfer Learning Analysis:")
            for agent_name, results in self.results.items():
                if 'transfer' in results:
                    source_perf = results['transfer']['source_performance']['mean_reward']
                    print(f"  {agent_name:20s}: Source performance {source_perf:.2f}")
                    for target_name, target_perf in results['transfer']['target_performances'].items():
                        transfer_ratio = target_perf['mean_reward'] / source_perf if source_perf > 0 else 0
                        print(f"    {target_name}: {target_perf['mean_reward']:.2f} (ratio: {transfer_ratio:.2f})")
        
        print("\n" + "="*80)
    
    def plot_results(self, save_path=None):
        """Create visualization plots of results"""
        if not self.results:
            print("No results to plot. Run comprehensive_evaluation() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sample efficiency plot
        ax = axes[0, 0]
        agent_names = []
        sample_efficiency = []
        for agent_name, results in self.results.items():
            if 'sample_efficiency' in results:
                agent_names.append(agent_name)
                sample_efficiency.append(results['sample_efficiency'])
        
        if agent_names:
            ax.bar(agent_names, sample_efficiency, color='steelblue', alpha=0.7)
            ax.set_title('Sample Efficiency Comparison', fontweight='bold')
            ax.set_ylabel('Episodes to Target')
            ax.tick_params(axis='x', rotation=45)
        
        # Performance comparison
        ax = axes[0, 1]
        all_rewards = {}
        for agent_name, results in self.results.items():
            rewards = []
            for key, value in results.items():
                if key.startswith('reward_env_'):
                    rewards.append(value['mean_reward'])
            if rewards:
                all_rewards[agent_name] = np.mean(rewards)
        
        if all_rewards:
            ax.bar(all_rewards.keys(), all_rewards.values(), color='seagreen', alpha=0.7)
            ax.set_title('Average Performance', fontweight='bold')
            ax.set_ylabel('Mean Reward')
            ax.tick_params(axis='x', rotation=45)
        
        # Transfer learning plot
        ax = axes[1, 0]
        transfer_ratios = {}
        for agent_name, results in self.results.items():
            if 'transfer' in results:
                source_perf = results['transfer']['source_performance']['mean_reward']
                target_ratios = []
                for target_perf in results['transfer']['target_performances'].values():
                    ratio = target_perf['mean_reward'] / source_perf if source_perf > 0 else 0
                    target_ratios.append(ratio)
                if target_ratios:
                    transfer_ratios[agent_name] = np.mean(target_ratios)
        
        if transfer_ratios:
            ax.bar(transfer_ratios.keys(), transfer_ratios.values(), color='coral', alpha=0.7)
            ax.set_title('Transfer Learning Capability', fontweight='bold')
            ax.set_ylabel('Transfer Ratio')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Transfer')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        
        # Performance variance
        ax = axes[1, 1]
        variances = {}
        for agent_name, results in self.results.items():
            stds = []
            for key, value in results.items():
                if key.startswith('reward_env_'):
                    stds.append(value['std_reward'])
            if stds:
                variances[agent_name] = np.mean(stds)
        
        if variances:
            ax.bar(variances.keys(), variances.values(), color='gold', alpha=0.7)
            ax.set_title('Performance Stability', fontweight='bold')
            ax.set_ylabel('Average Std Dev')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class IntegratedAdvancedAgent:
    """Integrated agent combining multiple advanced RL techniques"""

    def __init__(self, state_dim, action_dim, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Initialize components based on config
        self.use_prioritized_replay = self.config.get('use_prioritized_replay', True)
        self.use_auxiliary_tasks = self.config.get('use_auxiliary_tasks', True)
        self.use_data_augmentation = self.config.get('use_data_augmentation', True)
        self.use_world_model = self.config.get('use_world_model', False)
        self.use_hierarchical = self.config.get('use_hierarchical', False)
        
        # This would integrate multiple agent types
        # For now, return a simple placeholder
        self.performance_history = []

    def act(self, state, epsilon=0.1):
        """Select action"""
        # Placeholder implementation
        return np.random.randint(0, self.action_dim)
    
    def update(self):
        """Update agent parameters"""
        # Placeholder implementation
        pass
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience"""
        # Placeholder implementation
        pass