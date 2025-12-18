#!/usr/bin/env python3
"""
CA10: Model-Based Reinforcement Learning - Test Suite
====================================================

This script tests all components of the CA10 Model-Based RL project
to ensure everything is working correctly.

Author: DRL Course Team
Date: 2025
"""

import sys
import os
import importlib
import traceback
from typing import Dict, List, Tuple, Any
import json

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports() -> Dict[str, bool]:
    """Test all module imports"""
    print("🐍 Testing Python Imports...")

    import_results = {}

    # Test basic imports
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        import pandas as pd

        print("✅ Basic dependencies imported successfully")
        import_results["basic_deps"] = True
    except ImportError as e:
        print(f"❌ Basic dependencies import failed: {e}")
        import_results["basic_deps"] = False

    # Test project modules
    modules_to_test = [
        "models.models",
        "environments.environments",
        "agents.classical_planning",
        "agents.dyna_q",
        "agents.mcts",
        "agents.mpc",
        "experiments.comparison",
        "evaluation.evaluator",
        "evaluation.metrics",
        "utils.helpers",
        "utils.visualization",
    ]

    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
            import_results[module_name] = True
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            import_results[module_name] = False

    return import_results


def test_file_structure() -> Dict[str, bool]:
    """Test project file structure"""
    print("\n🔍 Testing Project Structure...")

    required_files = [
        "__init__.py",
        "README.md",
        "requirements.txt",
        "training_examples.py",
        "run.sh",
        "CA10.ipynb",
        "agents/__init__.py",
        "agents/classical_planning.py",
        "agents/dyna_q.py",
        "agents/mcts.py",
        "agents/mpc.py",
        "environments/__init__.py",
        "environments/environments.py",
        "models/__init__.py",
        "models/models.py",
        "experiments/__init__.py",
        "experiments/comparison.py",
        "evaluation/__init__.py",
        "evaluation/evaluator.py",
        "evaluation/metrics.py",
        "utils/__init__.py",
        "utils/helpers.py",
        "utils/visualization.py",
    ]

    required_dirs = [
        "agents",
        "environments",
        "models",
        "experiments",
        "evaluation",
        "utils",
        "visualizations",
        "CA10_files",
    ]

    structure_results = {}

    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            structure_results[file_path] = True
        else:
            print(f"❌ {file_path} - Missing")
            structure_results[file_path] = False

    # Check directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/")
            structure_results[f"{dir_path}/"] = True
        else:
            print(f"❌ {dir_path}/ - Missing")
            structure_results[f"{dir_path}/"] = False

    return structure_results


def test_basic_functionality() -> Dict[str, bool]:
    """Test basic functionality of key components"""
    print("\n🧪 Testing Basic Functionality...")

    functionality_results = {}

    try:
        # Test environment creation
        from environments.environments import SimpleGridWorld

        env = SimpleGridWorld(size=5)
        state = env.reset()
        action = 0
        next_state, reward, done = env.step(action)
        print("✅ Environment creation and interaction")
        functionality_results["environment"] = True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        functionality_results["environment"] = False

    try:
        # Test model creation
        from models.models import TabularModel

        model = TabularModel(num_states=25, num_actions=4)
        model.update(0, 1, 2, 0.5)
        prob = model.get_transition_prob(0, 1, 2)
        print("✅ Tabular model creation and updates")
        functionality_results["tabular_model"] = True
    except Exception as e:
        print(f"❌ Tabular model test failed: {e}")
        functionality_results["tabular_model"] = False

    try:
        # Test neural model creation
        from models.models import NeuralModel
        import torch

        model = NeuralModel(state_dim=25, action_dim=4, hidden_dim=32)
        state = torch.eye(25)[0:1]
        action = torch.tensor([0])
        next_state, reward = model(state, action)
        print("✅ Neural model creation and forward pass")
        functionality_results["neural_model"] = True
    except Exception as e:
        print(f"❌ Neural model test failed: {e}")
        functionality_results["neural_model"] = False

    try:
        # Test Dyna-Q agent
        from agents.dyna_q import DynaQAgent

        agent = DynaQAgent(num_states=25, num_actions=4, planning_steps=5)
        action = agent.select_action(0)
        agent.update(0, action, 0.5, 1, False)
        print("✅ Dyna-Q agent creation and updates")
        functionality_results["dyna_q"] = True
    except Exception as e:
        print(f"❌ Dyna-Q agent test failed: {e}")
        functionality_results["dyna_q"] = False

    try:
        # Test MCTS agent
        from agents.mcts import MCTSAgent
        from models.models import TabularModel

        model = TabularModel(25, 4)
        agent = MCTSAgent(model=model, num_states=25, num_actions=4, num_simulations=10)
        action = agent.select_action(0)
        print("✅ MCTS agent creation and action selection")
        functionality_results["mcts"] = True
    except Exception as e:
        print(f"❌ MCTS agent test failed: {e}")
        functionality_results["mcts"] = False

    try:
        # Test MPC agent
        from agents.mpc import MPCAgent
        from models.models import NeuralModel
        import torch

        model = NeuralModel(state_dim=25, action_dim=4, hidden_dim=32)
        agent = MPCAgent(model=model, num_states=25, num_actions=4, horizon=5)
        action = agent.controller.select_action(0)
        print("✅ MPC agent creation and action selection")
        functionality_results["mpc"] = True
    except Exception as e:
        print(f"❌ MPC agent test failed: {e}")
        functionality_results["mpc"] = False

    return functionality_results


def test_visualization_creation() -> Dict[str, bool]:
    """Test visualization creation"""
    print("\n📊 Testing Visualization Creation...")

    viz_results = {}

    try:
        # Test basic plotting
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")

        # Save to visualizations directory
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/test_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("✅ Basic plotting and saving")
        viz_results["basic_plotting"] = True
    except Exception as e:
        print(f"❌ Basic plotting test failed: {e}")
        viz_results["basic_plotting"] = False

    try:
        # Test utils visualization
        from utils.visualization import plot_learning_curves

        sample_results = {
            "Method1": {
                "episode_rewards": [0.1, 0.2, 0.3, 0.4, 0.5],
                "episode_lengths": [10, 12, 8, 15, 11],
            },
            "Method2": {
                "episode_rewards": [0.05, 0.15, 0.25, 0.35, 0.45],
                "episode_lengths": [12, 14, 10, 17, 13],
            },
        }

        fig = plot_learning_curves(
            sample_results, save_path="visualizations/test_learning_curves.png"
        )
        plt.close(fig)

        print("✅ Learning curves plotting")
        viz_results["learning_curves"] = True
    except Exception as e:
        print(f"❌ Learning curves test failed: {e}")
        viz_results["learning_curves"] = False

    return viz_results


def run_mini_experiment() -> Dict[str, bool]:
    """Run a mini experiment to test the full pipeline"""
    print("\n🚀 Running Mini Experiment...")

    experiment_results = {}

    try:
        from environments.environments import SimpleGridWorld
        from agents.dyna_q import DynaQAgent
        from models.models import TabularModel

        # Create environment
        env = SimpleGridWorld(size=4)

        # Create agent
        agent = DynaQAgent(num_states=16, num_actions=4, planning_steps=5)

        # Run a few episodes
        episode_rewards = []
        for episode in range(5):
            state = env.reset()
            total_reward = 0

            for step in range(20):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

            episode_rewards.append(total_reward)

        print(f"✅ Mini experiment completed - Episode rewards: {episode_rewards}")
        experiment_results["mini_experiment"] = True

    except Exception as e:
        print(f"❌ Mini experiment failed: {e}")
        traceback.print_exc()
        experiment_results["mini_experiment"] = False

    return experiment_results


def generate_test_report(all_results: Dict[str, Dict[str, bool]]) -> None:
    """Generate comprehensive test report"""
    print("\n📋 Generating Test Report...")

    # Calculate overall statistics
    total_tests = sum(
        len(category_results) for category_results in all_results.values()
    )
    passed_tests = sum(
        sum(category_results.values()) for category_results in all_results.values()
    )

    # Create report
    report = {
        "test_summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (
                (passed_tests / total_tests * 100) if total_tests > 0 else 0
            ),
        },
        "detailed_results": all_results,
        "recommendations": [],
    }

    # Add recommendations based on results
    if not all_results.get("imports", {}).get("basic_deps", False):
        report["recommendations"].append(
            "Install missing dependencies: pip install -r requirements.txt"
        )

    if not all_results.get("structure", {}).get("run.sh", False):
        report["recommendations"].append("Make run.sh executable: chmod +x run.sh")

    failed_imports = [
        name for name, result in all_results.get("imports", {}).items() if not result
    ]
    if failed_imports:
        report["recommendations"].append(
            f"Fix import issues in: {', '.join(failed_imports)}"
        )

    # Save report
    os.makedirs("results", exist_ok=True)
    with open("results/test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Test report saved to: results/test_report.json")

    # Print summary
    print(f"\n{'='*60}")
    print(f"🧪 CA10 Model-Based RL - Test Suite Results")
    print(f"{'='*60}")
    print(f"📊 Overall Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success Rate: {report['test_summary']['success_rate']:.1f}%")

    if report["recommendations"]:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

    if report["test_summary"]["success_rate"] >= 80:
        print(f"\n🎉 Project is ready to run!")
        print(f"Next steps:")
        print(f"1. Install dependencies: pip install -r requirements.txt")
        print(f"2. Run full project: ./run.sh")
        print(f"3. Or run specific components individually")
    else:
        print(f"\n⚠️ Some issues need to be resolved before running the full project.")

    print(f"{'='*60}")


def main():
    """Main test function"""
    print("🧪 CA10 Model-Based Reinforcement Learning - Test Suite")
    print("=" * 60)

    # Run all tests
    all_results = {}

    all_results["structure"] = test_file_structure()
    all_results["imports"] = test_imports()
    all_results["functionality"] = test_basic_functionality()
    all_results["visualization"] = test_visualization_creation()
    all_results["experiment"] = run_mini_experiment()

    # Generate report
    generate_test_report(all_results)

    return all_results


if __name__ == "__main__":
    results = main()
