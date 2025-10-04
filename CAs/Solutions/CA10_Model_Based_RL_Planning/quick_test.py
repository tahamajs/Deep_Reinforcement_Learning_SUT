#!/usr/bin/env python3
"""
Quick test for CA10 project structure without external dependencies
"""

import os
import sys


def test_basic_structure():
    """Test basic project structure"""
    print("🔍 Testing Basic Project Structure...")

    required_files = [
        "__init__.py",
        "README.md",
        "requirements.txt",
        "training_examples.py",
        "run.sh",
        "CA10.ipynb",
    ]

    required_dirs = [
        "agents",
        "environments",
        "models",
        "experiments",
        "evaluation",
        "utils",
        "visualizations",
    ]

    all_good = True

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_good = False

    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            all_good = False

    return all_good


def test_file_contents():
    """Test that key files have content"""
    print("\n📄 Testing File Contents...")

    files_to_check = {
        "README.md": 1000,  # At least 1000 characters
        "requirements.txt": 100,  # At least 100 characters
        "run.sh": 500,  # At least 500 characters
    }

    all_good = True

    for file_path, min_size in files_to_check.items():
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                if len(content) >= min_size:
                    print(
                        f"✅ {file_path} has substantial content ({len(content)} chars)"
                    )
                else:
                    print(f"⚠️ {file_path} has minimal content ({len(content)} chars)")
        else:
            print(f"❌ {file_path} not found")
            all_good = False

    return all_good


def test_agent_files():
    """Test agent files exist and have content"""
    print("\n🤖 Testing Agent Files...")

    agent_files = [
        "agents/__init__.py",
        "agents/classical_planning.py",
        "agents/dyna_q.py",
        "agents/mcts.py",
        "agents/mpc.py",
    ]

    all_good = True

    for file_path in agent_files:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                if len(content) >= 500:  # At least 500 characters
                    print(
                        f"✅ {file_path} has substantial content ({len(content)} chars)"
                    )
                else:
                    print(f"⚠️ {file_path} has minimal content ({len(content)} chars)")
        else:
            print(f"❌ {file_path} not found")
            all_good = False

    return all_good


def test_other_modules():
    """Test other module files"""
    print("\n📦 Testing Other Modules...")

    module_files = [
        "environments/environments.py",
        "models/models.py",
        "experiments/comparison.py",
        "evaluation/evaluator.py",
        "utils/helpers.py",
    ]

    all_good = True

    for file_path in module_files:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                if len(content) >= 500:
                    print(
                        f"✅ {file_path} has substantial content ({len(content)} chars)"
                    )
                else:
                    print(f"⚠️ {file_path} has minimal content ({len(content)} chars)")
        else:
            print(f"❌ {file_path} not found")
            all_good = False

    return all_good


def main():
    """Main test function"""
    print("🧪 CA10 Model-Based RL - Quick Structure Test")
    print("=" * 50)

    structure_ok = test_basic_structure()
    content_ok = test_file_contents()
    agents_ok = test_agent_files()
    modules_ok = test_other_modules()

    print("\n📊 Test Results:")
    print("=" * 30)
    print(f"Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Content: {'✅ PASS' if content_ok else '❌ FAIL'}")
    print(f"Agents: {'✅ PASS' if agents_ok else '❌ FAIL'}")
    print(f"Modules: {'✅ PASS' if modules_ok else '❌ FAIL'}")

    overall_pass = structure_ok and content_ok and agents_ok and modules_ok

    if overall_pass:
        print("\n🎉 All basic tests passed!")
        print("📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test: python3 test_ca10.py")
        print("3. Run the project: ./run.sh")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
