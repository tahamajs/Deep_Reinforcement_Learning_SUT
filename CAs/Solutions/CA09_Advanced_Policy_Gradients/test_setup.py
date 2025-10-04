#!/usr/bin/env python3
"""
Test script to verify CA9 setup and run basic functionality
"""

import sys
import os
import traceback


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import torch

        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import gymnasium as gym

        print("✅ Gymnasium imported successfully")
    except ImportError as e:
        print(f"❌ Gymnasium import failed: {e}")
        return False

    try:
        import matplotlib.pyplot as plt

        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False

    try:
        import numpy as np

        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    return True


def test_agent_imports():
    """Test if agent modules can be imported"""
    print("\nTesting agent imports...")

    try:
        from agents.reinforce import REINFORCEAgent, PolicyNetwork

        print("✅ REINFORCE agent imported successfully")
    except ImportError as e:
        print(f"❌ REINFORCE import failed: {e}")
        return False

    try:
        from agents.actor_critic import ActorCriticAgent

        print("✅ Actor-Critic agent imported successfully")
    except ImportError as e:
        print(f"❌ Actor-Critic import failed: {e}")
        return False

    try:
        from agents.ppo import PPOAgent

        print("✅ PPO agent imported successfully")
    except ImportError as e:
        print(f"❌ PPO import failed: {e}")
        return False

    try:
        from agents.continuous_control import ContinuousPPOAgent

        print("✅ Continuous Control agent imported successfully")
    except ImportError as e:
        print(f"❌ Continuous Control import failed: {e}")
        return False

    return True


def test_environment():
    """Test environment setup"""
    print("\nTesting environment setup...")

    try:
        import gymnasium as gym

        env = gym.make("CartPole-v1")
        state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        env.close()
        print("✅ CartPole environment working")
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_basic_training():
    """Test basic training functionality"""
    print("\nTesting basic training...")

    try:
        from agents.reinforce import REINFORCEAgent
        import gymnasium as gym

        env = gym.make("CartPole-v1")
        agent = REINFORCEAgent(
            state_dim=env.observation_space.shape[0], action_dim=env.action_space.n
        )

        # Run a few episodes
        for episode in range(3):
            state, _ = env.reset()
            total_reward = 0

            for step in range(100):
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.store_transition(state, action, reward, log_prob)
                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break

            agent.update()
            print(f"  Episode {episode + 1}: Reward = {total_reward}")

        env.close()
        print("✅ Basic training test passed")
        return True

    except Exception as e:
        print(f"❌ Basic training test failed: {e}")
        traceback.print_exc()
        return False


def test_visualizations():
    """Test visualization functionality"""
    print("\nTesting visualizations...")

    try:
        from utils.policy_gradient_visualizer import PolicyGradientVisualizer

        visualizer = PolicyGradientVisualizer()

        # Test basic visualization
        results = visualizer.demonstrate_policy_gradient_intuition()
        print("✅ Policy gradient visualizations working")
        return True

    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("CA9: Advanced Policy Gradient Methods - Setup Test")
    print("=" * 60)

    tests = [
        ("Basic Imports", test_imports),
        ("Agent Imports", test_agent_imports),
        ("Environment Setup", test_environment),
        ("Basic Training", test_basic_training),
        ("Visualizations", test_visualizations),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! CA9 setup is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

