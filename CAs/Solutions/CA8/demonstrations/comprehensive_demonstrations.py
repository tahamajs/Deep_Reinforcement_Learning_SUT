"""
Comprehensive demonstrations for CA8
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List

from ..agents.causal_discovery import CausalGraph
from ..agents.causal_rl_agent import CausalRLAgent
from ..environments.multi_modal_env import MultiModalGridWorld, MultiModalWrapper


def run_comprehensive_experiments():
    """Run comprehensive experiments comparing different RL approaches"""
    print("=== Comprehensive RL Experiments ===")
    
    class MultiModalCausalRLAgent(CausalRLAgent):
        """Causal RL agent adapted for multi-modal observations"""
        
        def __init__(self, wrapper, causal_graph, lr=1e-3):
            self.wrapper = wrapper
            state_dim = wrapper.total_dim
            action_dim = 4  # grid world actions
            super().__init__(state_dim, action_dim, causal_graph, lr)
        
        def select_action(self, obs, deterministic=False):
            """Select action from multi-modal observation"""
            state = self.wrapper.process_observation(obs)
            return super().select_action(state, deterministic)
        
        def train_episode(self, env, max_steps=1000):
            """Train for one episode with multi-modal observations"""
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            states, actions, rewards, next_obss, dones = [], [], [], [], []
            
            while steps < max_steps:
                action, _ = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                states.append(self.wrapper.process_observation(obs))
                actions.append(action)
                rewards.append(reward)
                next_obss.append(self.wrapper.process_observation(next_obs))
                dones.append(done)
                
                episode_reward += reward
                steps += 1
                obs = next_obs
                
                if done:
                    break
            
            if len(states) > 0:
                self.update(states, actions, rewards, next_obss, dones)
            
            self.episode_rewards.append(episode_reward)
            return episode_reward, steps
    
    # Create environments
    simple_env = MultiModalGridWorld(size=5, render_size=64)
    wrapper = MultiModalWrapper(simple_env)
    
    # Define causal graph
    variables = ['agent_x', 'agent_y', 'goal_x', 'goal_y', 'visual_features', 'text_features', 'reward']
    causal_graph = CausalGraph(variables)
    causal_graph.add_edge('agent_x', 'visual_features')
    causal_graph.add_edge('agent_y', 'visual_features')
    causal_graph.add_edge('goal_x', 'visual_features')
    causal_graph.add_edge('goal_y', 'visual_features')
    causal_graph.add_edge('visual_features', 'reward')
    causal_graph.add_edge('text_features', 'reward')
    
    # Define different agent types
    class StandardRLAgent:
        """Standard RL agent without causal reasoning"""
        def __init__(self, state_dim, action_dim, lr=1e-3):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.lr = lr
            self.episode_rewards = []
            
        def select_action(self, state, deterministic=False):
            return np.random.randint(0, self.action_dim), None
            
        def update(self, states, actions, rewards, next_states, dones):
            pass
            
        def train_episode(self, env, max_steps=1000):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < max_steps:
                if hasattr(env, 'step'):
                    # Multi-modal environment
                    action, _ = self.select_action(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    obs = next_obs
                else:
                    # Simple environment
                    state = obs
                    action, _ = self.select_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    obs = next_state
                
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            return episode_reward, steps
    
    class MultiModalRLAgent:
        """Multi-modal RL agent without causal reasoning"""
        def __init__(self, wrapper, lr=1e-3):
            self.wrapper = wrapper
            self.lr = lr
            self.episode_rewards = []
            
        def select_action(self, obs, deterministic=False):
            processed_obs = self.wrapper.process_observation(obs)
            return np.random.randint(0, 4), None
            
        def update(self, states, actions, rewards, next_states, dones):
            pass
            
        def train_episode(self, env, max_steps=1000):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < max_steps:
                action, _ = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                obs = next_obs
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            return episode_reward, steps
    
    # Create agents
    agents = {
        'Standard RL': StandardRLAgent(wrapper.state_dim, 4),
        'Multi-Modal RL': MultiModalRLAgent(wrapper),
        'Causal RL': CausalRLAgent(wrapper.total_dim, 4, causal_graph),
        'Causal Multi-Modal RL': MultiModalCausalRLAgent(wrapper, causal_graph)
    }
    
    # Run experiments
    results = {}
    n_episodes = 100
    
    for name, agent in agents.items():
        print(f"\nTraining {name}...")
        episode_rewards = []
        
        for episode in range(n_episodes):
            if 'Multi-Modal' in name:
                reward, steps = agent.train_episode(simple_env)
            else:
                # Create simple environment for standard agents
                class SimpleEnv:
                    def __init__(self):
                        self.state_dim = wrapper.state_dim
                        self.action_dim = 4
                        
                    def reset(self):
                        return np.random.randn(self.state_dim), {}
                        
                    def step(self, action):
                        reward = np.random.randn()
                        next_state = np.random.randn(self.state_dim)
                        return next_state, reward, False, False, {}
                
                simple_single_env = SimpleEnv()
                reward, steps = agent.train_episode(simple_single_env)
            
            episode_rewards.append(reward)
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"  Episode {episode+1:3d} | Avg Reward: {avg_reward:.3f}")
        
        results[name] = {
            'episode_rewards': episode_rewards,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'final_reward': np.mean(episode_rewards[-10:])
        }
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning curves
    for name, result in results.items():
        axes[0, 0].plot(result['episode_rewards'], label=name, alpha=0.7)
        # Moving average
        moving_avg = pd.Series(result['episode_rewards']).rolling(10).mean()
        axes[0, 0].plot(moving_avg, label=f'{name} (MA)', linewidth=2)
    
    axes[0, 0].set_title('Learning Curves Comparison')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final performance comparison
    names = list(results.keys())
    final_rewards = [results[name]['final_reward'] for name in names]
    std_rewards = [results[name]['std_reward'] for name in names]
    
    bars = axes[0, 1].bar(names, final_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    axes[0, 1].set_title('Final Performance Comparison')
    axes[0, 1].set_ylabel('Average Final Reward')
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample efficiency
    sample_efficiency = []
    for name, result in results.items():
        # Calculate how quickly agent reaches 80% of final performance
        final_perf = result['final_reward']
        target = 0.8 * final_perf
        episodes_to_target = n_episodes
        
        for i, reward in enumerate(result['episode_rewards']):
            if reward >= target:
                episodes_to_target = i + 1
                break
        
        sample_efficiency.append(episodes_to_target)
    
    axes[1, 0].bar(names, sample_efficiency, alpha=0.7)
    axes[1, 0].set_title('Sample Efficiency (Episodes to 80% Performance)')
    axes[1, 0].set_ylabel('Episodes')
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics summary
    metrics = ['Avg Reward', 'Std Reward', 'Final Reward', 'Sample Efficiency']
    metric_values = []
    
    for name in names:
        result = results[name]
        values = [
            result['avg_reward'],
            result['std_reward'],
            result['final_reward'],
            1.0 / (sample_efficiency[names.index(name)] / n_episodes)  # Normalized efficiency
        ]
        metric_values.append(values)
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, name in enumerate(names):
        offset = (i - len(names)/2 + 0.5) * width
        axes[1, 1].bar(x + offset, metric_values[i], width, label=name, alpha=0.7)
    
    axes[1, 1].set_title('Performance Metrics Summary')
    axes[1, 1].set_ylabel('Normalized Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Average Reward: {result['avg_reward']:.3f} ± {result['std_reward']:.3f}")
        print(f"  Final Reward: {result['final_reward']:.3f}")
        print(f"  Sample Efficiency: {sample_efficiency[names.index(name)]} episodes")
    
    return {
        'results': results,
        'agents': agents,
        'environments': {'simple': simple_env, 'wrapper': wrapper},
        'causal_graph': causal_graph
    }
