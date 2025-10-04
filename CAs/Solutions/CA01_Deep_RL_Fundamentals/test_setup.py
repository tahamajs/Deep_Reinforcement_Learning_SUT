#!/usr/bin/env python3
"""
Test script to verify that all components are working correctly.
"""

import sys
import os
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        # Test agent imports
        from agents.ca1_agents import DQNAgent, REINFORCEAgent, ActorCriticAgent

        print("✓ Agents imported successfully")

        # Test model imports
        from models.ca1_models import DQN, DuelingDQN, NoisyDQN

        print("✓ Models imported successfully")

        # Test utility imports
        from utils.ca1_utils import set_seed, moving_average, device

        print("✓ Utils imported successfully")

        # Test environment imports
        from environments.custom_envs import (
            SimpleGridWorld,
            MultiArmedBandit,
            create_cartpole_env,
        )

        print("✓ Environments imported successfully")

        # Test evaluation imports
        from evaluation.evaluators import AgentEvaluator, compare_agents

        print("✓ Evaluators imported successfully")

        # Test experiment imports
        from experiments.experiment_runner import ExperimentRunner

        print("✓ Experiment runner imported successfully")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test that agents can be created."""
    print("\nTesting agent creation...")

    try:
        from agents.ca1_agents import DQNAgent, REINFORCEAgent, ActorCriticAgent
        from utils.ca1_utils import set_seed

        set_seed(42)

        # Test DQN agent
        dqn_agent = DQNAgent(
            state_size=4, action_size=2, use_dueling=True, use_double_dqn=True
        )
        print("✓ DQN agent created successfully")

        # Test REINFORCE agent
        reinforce_agent = REINFORCEAgent(state_size=4, action_size=2)
        print("✓ REINFORCE agent created successfully")

        # Test Actor-Critic agent
        ac_agent = ActorCriticAgent(state_size=4, action_size=2)
        print("✓ Actor-Critic agent created successfully")

        return True

    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test that environments can be created."""
    print("\nTesting environment creation...")

    try:
        from environments.custom_envs import (
            SimpleGridWorld,
            MultiArmedBandit,
            create_cartpole_env,
        )

        # Test custom environments
        grid_world = SimpleGridWorld(grid_size=5)
        print("✓ SimpleGridWorld created successfully")

        bandit = MultiArmedBandit(n_arms=5)
        print("✓ MultiArmedBandit created successfully")

        # Test gymnasium environment
        cartpole_env = create_cartpole_env()
        print("✓ CartPole environment created successfully")
        cartpole_env.close()

        return True

    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of agents and environments."""
    print("\nTesting basic functionality...")

    try:
        from agents.ca1_agents import DQNAgent
        from environments.custom_envs import create_cartpole_env
        from utils.ca1_utils import set_seed
        import numpy as np

        set_seed(42)

        # Create environment and agent
        env = create_cartpole_env()
        agent = DQNAgent(state_size=4, action_size=2)

        # Test one episode
        state, _ = env.reset()
        total_reward = 0

        for step in range(10):  # Short test
            action = agent.act(state)
            result = env.step(action)

            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        env.close()
        print(f"✓ Basic functionality test passed (reward: {total_reward:.2f})")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_visualization_setup():
    """Test that visualization libraries are available."""
    print("\nTesting visualization setup...")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Test basic plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")

        # Save test plot
        os.makedirs("visualization", exist_ok=True)
        plt.savefig("visualization/test_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("✓ Visualization setup successful")
        return True

    except Exception as e:
        print(f"✗ Visualization setup failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("CA1 Deep RL Fundamentals - Setup Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_agent_creation,
        test_environment_creation,
        test_basic_functionality,
        test_visualization_setup,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("🎉 All tests passed! The setup is ready.")
        print("\nYou can now run:")
        print("  ./run.sh quick    # Quick execution")
        print("  ./run.sh full     # Full execution")
        print("  ./run.sh help     # Show all options")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


