"""
Multi-modal demonstrations for CA8
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any

from ..agents.causal_discovery import CausalGraph
from ..agents.causal_rl_agent import CausalRLAgent
from ..environments.multi_modal_env import MultiModalGridWorld, MultiModalWrapper


def demonstrate_multi_modal_env():
    """Demonstrate multi-modal environment and wrapper"""
    print("=== Multi-Modal Environment Demonstration ===")
    
    # Create environment
    env = MultiModalGridWorld(size=6, render_size=84, max_steps=100)
    wrapper = MultiModalWrapper(env)
    
    print(f"Environment size: {env.size}x{env.size}")
    print(f"Visual observation shape: {env.render_size}x{env.render_size}x3")
    print(f"Text observation keys: {list(env.text_template.keys())}")
    print(f"State observation keys: {list(env.state_template.keys())}")
    print(f"Wrapper total dimension: {wrapper.total_dim}")
    
    # Test environment
    obs, info = env.reset()
    print(f"\nInitial observation keys: {list(obs.keys())}")
    print(f"Visual shape: {obs['visual'].shape}")
    print(f"Text: {obs['text']}")
    print(f"State: {obs['state']}")
    
    # Test wrapper
    processed_obs = wrapper.process_observation(obs)
    print(f"Processed observation shape: {processed_obs.shape}")
    
    # Test a few steps
    print("\nTesting environment steps...")
    for step in range(3):
        action = np.random.randint(0, 4)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Step {step+1}: Action {action}, Reward {reward:.3f}, Done {done}")
        if done:
            break
    
    return {
        'environment': env,
        'wrapper': wrapper,
        'initial_obs': obs,
        'processed_obs': processed_obs
    }


def demonstrate_integrated_system():
    """Demonstrate integrated causal multi-modal RL system"""
    print("=== Integrated Causal Multi-Modal RL Demonstration ===")
    
    # Create environment and wrapper
    env = MultiModalGridWorld(size=6, render_size=64, max_steps=100)
    wrapper = MultiModalWrapper(env)
    
    # Define causal graph for multi-modal RL
    variables = ['agent_x', 'agent_y', 'goal_x', 'goal_y', 'visual_features', 'text_features', 'reward']
    causal_graph = CausalGraph(variables)
    
    causal_graph.add_edge('agent_x', 'visual_features')
    causal_graph.add_edge('agent_y', 'visual_features')
    causal_graph.add_edge('goal_x', 'visual_features')
    causal_graph.add_edge('goal_y', 'visual_features')
    causal_graph.add_edge('agent_x', 'text_features')
    causal_graph.add_edge('agent_y', 'text_features')
    causal_graph.add_edge('goal_x', 'text_features')
    causal_graph.add_edge('goal_y', 'text_features')
    causal_graph.add_edge('visual_features', 'reward')
    causal_graph.add_edge('text_features', 'reward')
    
    print(f"Causal graph for multi-modal RL: {causal_graph}")
    
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
        
        def train_episode(self, env):
            """Train for one episode with multi-modal observations"""
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            states, actions, rewards, next_obss, dones = [], [], [], [], []
            
            while steps < env.max_steps:
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
    
    # Create and train agent
    agent = MultiModalCausalRLAgent(wrapper, causal_graph, lr=1e-3)
    
    print("\nTraining Multi-Modal Causal RL Agent...")
    training_rewards = []
    
    for episode in range(50):  # Shorter training for demo
        reward, steps = agent.train_episode(env)
        training_rewards.append(reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_rewards[-10:])
            print(f"Episode {episode+1:2d} | Avg Reward: {avg_reward:.3f} | Steps: {steps}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(training_rewards)
    axes[0].plot(pd.Series(training_rewards).rolling(5).mean(), 
                 color='red', label='Moving Average')
    axes[0].set_title('Multi-Modal Causal RL Training')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    obs, _ = env.reset()
    axes[1].imshow(obs['visual'])
    axes[1].set_title(f'Environment Render\n{obs["text"]["text"]}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'agent': agent,
        'environment': env,
        'wrapper': wrapper,
        'training_rewards': training_rewards,
        'causal_graph': causal_graph
    }
