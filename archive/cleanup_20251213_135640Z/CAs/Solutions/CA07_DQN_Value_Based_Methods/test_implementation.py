#!/usr/bin/env python3
"""
Test script for CA07 DQN implementation
=========================================
This script tests the basic functionality of the DQN implementation
"""

import sys
import os
import torch
import numpy as np
import gymnasium as gym
import pytest
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, NoisyDQNAgent # Add NoisyDQNAgent
from src.utils import set_seed
from src.config import DQNConfig

# Initialize config
config = DQNConfig()

def test_basic_dqn():
    """Test basic DQN functionality"""
    print("Testing Basic DQN...")

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Create environment
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

    # Test action selection
    state, _ = env.reset(seed=config.SEED)
    action = agent.select_action(state)
    assert 0 <= action < action_dim, f"Invalid action: {action}"

    # Test training step
    for _ in range(10):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        # Assert loss is not None only if buffer is full enough to sample
        if len(agent.replay_buffer) >= config.BATCH_SIZE:
            assert loss is not None

        state = next_state
        if done:
            state, _ = env.reset(seed=config.SEED)

    # Test evaluation
    eval_results = agent.evaluate(env, num_episodes=2, max_steps=config.EVAL_MAX_STEPS)
    assert "mean_reward" in eval_results
    assert eval_results["mean_reward"] >= -500 # CartPole minimal score

    env.close()
    print("✓ Basic DQN test passed!")


def test_double_dqn():
    """Test Double DQN functionality"""
    print("Testing Double DQN...")

    set_seed(config.SEED)

    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

    # Test training
    state, _ = env.reset(seed=config.SEED)
    for _ in range(10):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        if len(agent.replay_buffer) >= config.BATCH_SIZE:
            assert loss is not None

        state = next_state
        if done:
            state, _ = env.reset(seed=config.SEED)

    # Test bias analysis (if implemented in agent)
    # Note: analyze_overestimation_bias is in DoubleDQNAgent in training_examples.py,
    # but for modularity, it should be in src/agents.py or a specific evaluation module.
    # Assuming it's in DoubleDQNAgent for now.
    if hasattr(agent, 'analyze_overestimation_bias'):
        bias_stats = agent.analyze_overestimation_bias(env, num_samples=10)
        assert "mean_bias" in bias_stats
    else:
        print("  Skipping bias analysis: analyze_overestimation_bias not found in agent.")

    env.close()
    print("✓ Double DQN test passed!")


def test_dueling_dqn():
    """Test Dueling DQN functionality"""
    print("Testing Dueling DQN...")

    set_seed(config.SEED)

    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

    # Test training
    state, _ = env.reset(seed=config.SEED)
    for _ in range(10):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        if len(agent.replay_buffer) >= config.BATCH_SIZE:
            assert loss is not None

        state = next_state
        if done:
            state, _ = env.reset(seed=config.SEED)

    # Test value-advantage decomposition (if implemented in agent)
    if hasattr(agent, 'get_value_and_advantage'):
        state, _ = env.reset(seed=config.SEED)
        value, advantage = agent.get_value_and_advantage(state)
        assert isinstance(value, float)
        assert len(advantage) == action_dim

    # Test decomposition analysis (if implemented in agent)
    if hasattr(agent, 'analyze_value_advantage_decomposition'):
        decomp_stats = agent.analyze_value_advantage_decomposition(env, num_samples=10)
        assert "mean_value" in decomp_stats
        assert "mean_advantage" in decomp_stats
    else:
        print("  Skipping decomposition analysis: analyze_value_advantage_decomposition not found in agent.")

    env.close()
    print("✓ Dueling DQN test passed!")

def test_noisy_dqn():
    """Test Noisy DQN functionality"""
    print("Testing Noisy DQN...")

    set_seed(config.SEED)

    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = NoisyDQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)

    # Test training
    state, _ = env.reset(seed=config.SEED)
    for _ in range(10):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.train_step()
        if len(agent.replay_buffer) >= config.BATCH_SIZE:
            assert loss is not None

        state = next_state
        if done:
            state, _ = env.reset(seed=config.SEED)

    env.close()
    print("✓ Noisy DQN test passed!")

def test_utilities():
    """Test utility functions"""
    print("Testing utilities...")

    # Test set_seed
    set_seed(100)
    rand1 = random.random()
    set_seed(100)
    rand2 = random.random()
    assert rand1 == rand2, "set_seed failed for random module"

    np_rand1 = np.random.rand(1)
    set_seed(100)
    np_rand2 = np.random.rand(1)
    assert np_rand1 == np_rand2, "set_seed failed for numpy"

    torch_rand1 = torch.rand(1)
    set_seed(100)
    torch_rand2 = torch.rand(1)
    assert torch_rand1 == torch_rand2, "set_seed failed for torch"

    # Test smooth_curve
    from src.utils import smooth_curve # Import after set_seed to ensure it's from src
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    smoothed = smooth_curve(data, window_size=3)
    expected_smoothed = np.array([2., 3., 4., 5., 6., 7., 8., 9.])
    assert np.allclose(smoothed, expected_smoothed), f"smooth_curve failed: Expected {expected_smoothed}, got {smoothed}"

    print("✓ Utilities test passed!")


def run_quick_training_test():
    """Run a quick training test"""
    print("Running quick training test...")

    set_seed(config.SEED)

    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Test different agents
    agents = {
        "DQN": DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config),
        "Double DQN": DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, config=config
        ),
        "Dueling DQN": DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, config=config
        ),
        "Noisy DQN": NoisyDQNAgent(
            state_dim=state_dim, action_dim=action_dim, config=config
        ),
    }

    results = {}

    for name, agent in agents.items():
        print(f"  Training {name}...")
        scores = []

        for episode in range(10):  # Short training for quick test
            reward, _ = agent.train_episode(env, max_steps=config.MAX_EPISODE_STEPS)
            scores.append(reward)

        results[name] = {
            "scores": scores,
            "final_score": np.mean(scores[-3:]), # Average last 3 episodes
            "max_score": np.max(scores),
        }

        print(f"    Final score: {results[name]['final_score']:.2f}")
        print(f"    Max score: {results[name]['max_score']:.2f}")

    env.close()

    # Find best agent
    best_agent = max(results.keys(), key=lambda x: results[x]["final_score"])
    print(f"\nBest agent in quick test: {best_agent}")
    print("✓ Quick training test completed!")

def main():
    """Run all tests"""
    print("CA07 DQN Implementation Tests")
    print("=" * 40)

    try:
        test_basic_dqn()
        test_double_dqn()
        test_dueling_dqn()
        test_noisy_dqn() # Add test for NoisyDQN
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


