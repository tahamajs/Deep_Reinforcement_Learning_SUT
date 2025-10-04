#!/usr/bin/env python3
"""
Test script for CA07 DQN implementation
=========================================
This script tests the basic functionality of the DQN implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import gymnasium as gym
from agents.core import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from utils import set_seed, calculate_statistics

def test_basic_dqn():
    """Test basic DQN functionality"""
    print("Testing Basic DQN...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state)
    assert 0 <= action < action_dim, f"Invalid action: {action}"
    
    # Test training step
    for _ in range(100):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        
        state = next_state
        if done:
            state, _ = env.reset()
    
    # Test evaluation
    eval_results = agent.evaluate(env, num_episodes=5)
    assert 'mean_reward' in eval_results
    assert eval_results['mean_reward'] >= 0
    
    env.close()
    print("✓ Basic DQN test passed!")

def test_double_dqn():
    """Test Double DQN functionality"""
    print("Testing Double DQN...")
    
    set_seed(42)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)
    
    # Test training
    state, _ = env.reset()
    for _ in range(100):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        
        state = next_state
        if done:
            state, _ = env.reset()
    
    # Test bias analysis
    bias_stats = agent.analyze_overestimation_bias(env, num_samples=50)
    assert 'mean_bias' in bias_stats
    
    env.close()
    print("✓ Double DQN test passed!")

def test_dueling_dqn():
    """Test Dueling DQN functionality"""
    print("Testing Dueling DQN...")
    
    set_seed(42)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DuelingDQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)
    
    # Test training
    state, _ = env.reset()
    for _ in range(100):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        
        state = next_state
        if done:
            state, _ = env.reset()
    
    # Test value-advantage decomposition
    state, _ = env.reset()
    value, advantage = agent.get_value_and_advantage(state)
    assert isinstance(value, float)
    assert len(advantage) == action_dim
    
    # Test decomposition analysis
    decomp_stats = agent.analyze_value_advantage_decomposition(env, num_samples=50)
    assert 'mean_value' in decomp_stats
    assert 'mean_advantage' in decomp_stats
    
    env.close()
    print("✓ Dueling DQN test passed!")

def test_utilities():
    """Test utility functions"""
    print("Testing utilities...")
    
    # Test statistics calculation
    data = [1, 2, 3, 4, 5]
    stats = calculate_statistics(data)
    assert stats['mean'] == 3.0
    assert stats['min'] == 1.0
    assert stats['max'] == 5.0
    
    print("✓ Utilities test passed!")

def run_quick_training_test():
    """Run a quick training test"""
    print("Running quick training test...")
    
    set_seed(42)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Test different agents
    agents = {
        'DQN': DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3),
        'Double DQN': DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3),
        'Dueling DQN': DuelingDQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)
    }
    
    results = {}
    
    for name, agent in agents.items():
        print(f"  Training {name}...")
        scores = []
        
        for episode in range(50):  # Short training
            reward, _ = agent.train_episode(env, max_steps=200)
            scores.append(reward)
        
        results[name] = {
            'scores': scores,
            'final_score': np.mean(scores[-10:]),
            'max_score': np.max(scores)
        }
        
        print(f"    Final score: {results[name]['final_score']:.2f}")
        print(f"    Max score: {results[name]['max_score']:.2f}")
    
    env.close()
    
    # Find best agent
    best_agent = max(results.keys(), key=lambda x: results[x]['final_score'])
    print(f"\nBest agent: {best_agent}")
    print("✓ Quick training test completed!")

def main():
    """Run all tests"""
    print("CA07 DQN Implementation Tests")
    print("=" * 40)
    
    try:
        test_basic_dqn()
        test_double_dqn()
        test_dueling_dqn()
        test_utilities()
        run_quick_training_test()
        
        print("\n" + "=" * 40)
        print("All tests passed! ✓")
        print("=" * 40)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
