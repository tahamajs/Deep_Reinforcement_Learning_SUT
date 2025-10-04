#!/usr/bin/env python3
"""
Test script to verify all components are working correctly
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")

    try:
        from environments.environments import GridWorld, create_custom_environment

        print("✓ Environments module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import environments: {e}")
        return False

    try:
        from agents.policies import (
            RandomPolicy,
            CustomPolicy,
            GreedyPolicy,
            create_policy,
        )

        print("✓ Policies module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import policies: {e}")
        return False

    try:
        from agents.algorithms import (
            policy_evaluation,
            policy_iteration,
            value_iteration,
            q_learning,
        )

        print("✓ Algorithms module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import algorithms: {e}")
        return False

    try:
        from utils.visualization import (
            plot_value_function,
            plot_policy,
            plot_learning_curve,
        )

        print("✓ Visualization module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import visualization: {e}")
        return False

    try:
        from experiments.experiments import run_all_experiments

        print("✓ Experiments module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import experiments: {e}")
        return False

    try:
        from evaluation.metrics import (
            evaluate_policy_performance,
            compare_algorithm_convergence,
        )

        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import evaluation: {e}")
        return False

    try:
        from models import ModelManager, create_model_from_policy

        print("✓ Models module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of each module"""
    print("\n🔧 Testing basic functionality...")

    try:
        from environments.environments import GridWorld
        from agents.policies import RandomPolicy
        from agents.algorithms import policy_evaluation

        # Create environment and policy
        env = GridWorld()
        policy = RandomPolicy(env)

        # Test policy evaluation
        values = policy_evaluation(env, policy, gamma=0.9)
        print(
            f"✓ Policy evaluation works - start state value: {values[env.start_state]:.3f}"
        )

        # Test environment properties
        print(
            f"✓ Environment created - size: {env.size}x{env.size}, states: {len(env.states)}"
        )

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_visualization():
    """Test visualization functions"""
    print("\n📊 Testing visualization...")

    try:
        from environments.environments import GridWorld
        from agents.policies import RandomPolicy
        from agents.algorithms import policy_evaluation
        from utils.visualization import plot_value_function
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        env = GridWorld()
        policy = RandomPolicy(env)
        values = policy_evaluation(env, policy, gamma=0.9)

        # Test plotting (without showing)
        plot_value_function(env, values, "Test Value Function")
        plt.savefig("test_plot.png", dpi=100, bbox_inches="tight")
        plt.close()

        # Clean up test file
        if os.path.exists("test_plot.png"):
            os.remove("test_plot.png")

        print("✓ Visualization functions work correctly")
        return True

    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def test_models():
    """Test model creation and management"""
    print("\n💾 Testing model management...")

    try:
        from environments.environments import GridWorld
        from agents.policies import RandomPolicy
        from models import ModelManager, create_model_from_policy

        env = GridWorld()
        policy = RandomPolicy(env)

        # Create model manager
        manager = ModelManager("test_models")

        # Create and save a policy model
        policy_model = create_model_from_policy(policy, env, "test_policy")
        manager.save_model(policy_model, "test", "policy")

        # Load the model back
        loaded_model = manager.load_model("test", "policy")

        # Test model functionality
        action = loaded_model.get_action(env.start_state)
        print(f"✓ Model creation and loading works - sample action: {action}")

        # Clean up
        import shutil

        if os.path.exists("test_models"):
            shutil.rmtree("test_models")

        return True

    except Exception as e:
        print(f"❌ Model management test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting RL GridWorld Dynamic Programming Installation Test")
    print("=" * 60)

    all_tests_passed = True

    # Run tests
    if not test_imports():
        all_tests_passed = False

    if not test_basic_functionality():
        all_tests_passed = False

    if not test_visualization():
        all_tests_passed = False

    if not test_models():
        all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! The installation is working correctly.")
        print("✅ You can now run './run.sh' to execute all experiments.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

