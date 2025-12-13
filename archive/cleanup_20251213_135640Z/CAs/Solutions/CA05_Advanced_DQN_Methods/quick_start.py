#!/usr/bin/env python3
"""
CA5 Advanced DQN Methods - Quick Start Guide
This script provides a quick way to get started with the project
"""

import os
import sys
import subprocess
import json
from datetime import datetime


def print_banner():
    """Print project banner"""
    print("=" * 70)
    print("🚀 CA5 Advanced DQN Methods - Quick Start")
    print("=" * 70)
    print("Complete implementation of advanced DQN methods including:")
    print("• Vanilla DQN")
    print("• Double DQN")
    print("• Dueling DQN")
    print("• Prioritized Experience Replay")
    print("• Rainbow DQN")
    print("=" * 70)


def check_requirements():
    """Check if basic requirements are met"""
    print("\n🔍 Checking Requirements...")

    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    else:
        print(
            f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}"
        )
        return False

    # Check if we're in the right directory
    current_dir = os.getcwd()
    if "CA05_Advanced_DQN_Methods" in current_dir:
        print(f"✅ In correct directory: {current_dir}")
    else:
        print(f"❌ Please run from CA05_Advanced_DQN_Methods directory")
        return False

    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing Dependencies...")

    try:
        # Try to install requirements
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


def run_basic_test():
    """Run a basic test to verify installation"""
    print("\n🧪 Running Basic Test...")

    try:
        # Test environment creation
        test_code = """
import sys
sys.path.append('.')
from environments.custom_envs import GridWorldEnv
import numpy as np

print("Testing GridWorld environment...")
env = GridWorldEnv(size=3)
state = env.reset()
print(f"Initial state shape: {state.shape}")

for i in range(3):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(f"Step {i}: Action={action}, Reward={reward:.2f}, Done={done}")
    if done:
        break

env.close()
print("✅ Basic test completed successfully!")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print("✅ Basic test passed")
            print(result.stdout)
            return True
        else:
            print(f"❌ Basic test failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    print("-" * 50)

    examples = [
        {
            "title": "Run Complete Project",
            "command": "./run.sh",
            "description": "Executes all components and generates results",
        },
        {
            "title": "Train Single Agent",
            "command": "python main.py --mode train --agent dqn --episodes 1000",
            "description": "Train a DQN agent for 1000 episodes",
        },
        {
            "title": "Compare All Agents",
            "command": "python main.py --mode compare --episodes 500",
            "description": "Compare all DQN variants",
        },
        {
            "title": "Run Experiments",
            "command": "python main.py --mode experiment --env CartPole-v1",
            "description": "Run structured experiments",
        },
        {
            "title": "Evaluate Performance",
            "command": "python main.py --mode evaluate --episodes 100",
            "description": "Evaluate trained agents",
        },
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")
        print()


def show_project_structure():
    """Show project structure"""
    print("\n📁 Project Structure:")
    print("-" * 50)

    structure = """
CA05_Advanced_DQN_Methods/
├── agents/                 # DQN agent implementations
│   ├── dqn_base.py        # Base DQN
│   ├── double_dqn.py      # Double DQN
│   ├── dueling_dqn.py     # Dueling DQN
│   ├── prioritized_replay.py # Prioritized DQN
│   └── rainbow_dqn.py     # Rainbow DQN
├── environments/          # Custom environments
│   └── custom_envs.py     # GridWorld, etc.
├── utils/                 # Utility functions
│   ├── advanced_dqn_extensions.py
│   ├── network_architectures.py
│   ├── training_analysis.py
│   ├── analysis_tools.py
│   ├── ca5_helpers.py
│   └── ca5_main.py
├── experiments/           # Experiment configurations
├── evaluation/           # Performance evaluation
├── visualizations/        # Generated plots
├── models/               # Saved models
├── results/              # Results and reports
├── training_examples.py  # Training examples
├── CA5.ipynb            # Jupyter notebook
├── run.sh               # Complete execution script
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
"""

    print(structure)


def generate_quick_start_report():
    """Generate a quick start report"""
    print("\n📊 Generating Quick Start Report...")

    report = {
        "timestamp": datetime.now().isoformat(),
        "project": "CA5 Advanced DQN Methods",
        "status": "Ready for execution",
        "components": [
            "Vanilla DQN",
            "Double DQN",
            "Dueling DQN",
            "Prioritized Experience Replay",
            "Rainbow DQN",
        ],
        "environments": [
            "CartPole-v1",
            "MountainCar-v0",
            "LunarLander-v2",
            "GridWorld (custom)",
        ],
        "execution_options": [
            "Complete run: ./run.sh",
            "Training: python main.py --mode train",
            "Comparison: python main.py --mode compare",
            "Experiments: python main.py --mode experiment",
            "Evaluation: python main.py --mode evaluate",
        ],
        "output_directories": [
            "visualizations/ - Generated plots",
            "results/ - Training results",
            "experiments/ - Experiment results",
            "models/ - Saved models",
        ],
    }

    # Save report
    with open("results/quick_start_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Quick start report saved to results/quick_start_report.json")


def main():
    """Main quick start function"""

    print_banner()

    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return

    # Show project structure
    show_project_structure()

    # Show usage examples
    show_usage_examples()

    # Ask if user wants to install dependencies
    print("🔧 Setup Options:")
    print("1. Install dependencies and run basic test")
    print("2. Skip installation and show usage only")

    try:
        choice = input("\nEnter your choice (1 or 2): ").strip()

        if choice == "1":
            if install_dependencies():
                run_basic_test()
            else:
                print("❌ Installation failed. You can still use the project manually.")
        elif choice == "2":
            print("✅ Skipping installation. You can install dependencies later with:")
            print("   pip install -r requirements.txt")
        else:
            print("❌ Invalid choice. Showing usage only.")

    except KeyboardInterrupt:
        print("\n\n⏹️ Setup interrupted by user.")

    # Generate report
    generate_quick_start_report()

    print("\n" + "=" * 70)
    print("🎉 CA5 Advanced DQN Methods is ready!")
    print("=" * 70)
    print("Next steps:")
    print("1. Run complete project: ./run.sh")
    print("2. Or run specific mode: python main.py --mode train")
    print("3. Check results in visualizations/ and results/ folders")
    print("=" * 70)


if __name__ == "__main__":
    main()


