#!/usr/bin/env python3
"""
CA14 Test Script - Quick validation of all modules
"""

import sys
import os
import traceback

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)


def test_imports():
    """Test all module imports."""
    print("🧪 Testing module imports...")

    try:
        # Test offline RL
        from offline_rl import (
            ConservativeQLearning,
            ImplicitQLearning,
            generate_offline_dataset,
        )

        print("✅ Offline RL modules imported successfully")

        # Test safe RL
        from safe_rl import (
            SafeEnvironment,
            ConstrainedPolicyOptimization,
            LagrangianSafeRL,
        )

        print("✅ Safe RL modules imported successfully")

        # Test multi-agent RL
        from multi_agent import MultiAgentEnvironment, MADDPGAgent, QMIXAgent

        print("✅ Multi-Agent RL modules imported successfully")

        # Test robust RL
        from robust_rl import (
            RobustEnvironment,
            DomainRandomizationAgent,
            AdversarialRobustAgent,
        )

        print("✅ Robust RL modules imported successfully")

        # Test evaluation
        from evaluation import ComprehensiveEvaluator

        print("✅ Evaluation modules imported successfully")

        # Test utils
        from utils import create_evaluation_environments, run_comprehensive_evaluation

        print("✅ Utils modules imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test agent creation."""
    print("\n🤖 Testing agent creation...")

    try:
        from offline_rl import ConservativeQLearning, ImplicitQLearning
        from safe_rl import ConstrainedPolicyOptimization, LagrangianSafeRL
        from multi_agent import MADDPGAgent, QMIXAgent
        from robust_rl import DomainRandomizationAgent, AdversarialRobustAgent

        # Create agents
        cql = ConservativeQLearning(state_dim=2, action_dim=4)
        iql = ImplicitQLearning(state_dim=2, action_dim=4)
        cpo = ConstrainedPolicyOptimization(state_dim=2, action_dim=4)
        lagrangian = LagrangianSafeRL(state_dim=2, action_dim=4)
        maddpg = MADDPGAgent(obs_dim=6, action_dim=5, num_agents=3, agent_id=0)
        qmix = QMIXAgent(obs_dim=6, action_dim=5, num_agents=3, state_dim=18)
        dr_agent = DomainRandomizationAgent(obs_dim=6, action_dim=4)
        adv_agent = AdversarialRobustAgent(obs_dim=6, action_dim=4)

        print("✅ All agents created successfully")
        return True

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test environment creation."""
    print("\n🌍 Testing environment creation...")

    try:
        from safe_rl import SafeEnvironment
        from multi_agent import MultiAgentEnvironment
        from robust_rl import RobustEnvironment

        # Create environments
        safe_env = SafeEnvironment(size=6)
        ma_env = MultiAgentEnvironment(grid_size=6, num_agents=3, num_targets=2)
        robust_env = RobustEnvironment(base_size=6, uncertainty_level=0.1)

        print("✅ All environments created successfully")
        return True

    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_generation():
    """Test dataset generation."""
    print("\n📊 Testing dataset generation...")

    try:
        from offline_rl import generate_offline_dataset

        # Generate datasets
        expert_dataset = generate_offline_dataset(dataset_type="expert", size=1000)
        mixed_dataset = generate_offline_dataset(dataset_type="mixed", size=1000)
        random_dataset = generate_offline_dataset(dataset_type="random", size=1000)

        print(f"✅ Expert dataset: {len(expert_dataset)} samples")
        print(f"✅ Mixed dataset: {len(mixed_dataset)} samples")
        print(f"✅ Random dataset: {len(random_dataset)} samples")

        return True

    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation framework."""
    print("\n📈 Testing evaluation framework...")

    try:
        from evaluation import ComprehensiveEvaluator
        import numpy as np

        evaluator = ComprehensiveEvaluator()

        # Test with sample data
        training_curves = {
            "CQL": np.random.random(50) * 10,
            "IQL": np.random.random(50) * 10,
            "CPO": np.random.random(50) * 10,
        }

        efficiency_scores = evaluator.evaluate_sample_efficiency(training_curves)
        asymptotic_scores = evaluator.evaluate_asymptotic_performance(training_curves)

        print("✅ Evaluation framework tested successfully")
        return True

    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 CA14 Module Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_agent_creation,
        test_environment_creation,
        test_dataset_generation,
        test_evaluation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! CA14 is ready to run.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


