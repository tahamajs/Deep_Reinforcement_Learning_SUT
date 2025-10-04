#!/usr/bin/env python3
"""
CA14 Quick Start Script
اجرای سریع پروژه CA14 برای یادگیری تقویتی پیشرفته
"""

import sys
import os
import subprocess


def main():
    print("🚀 CA14 Advanced Deep Reinforcement Learning - Quick Start")
    print("=" * 60)
    print()

    # Check if we're in the right directory
    if not os.path.exists("run.sh"):
        print("❌ Please run this script from the CA14 project directory")
        print("   Expected files: run.sh, training_examples.py, CA14.ipynb")
        return 1

    print("📋 Available options:")
    print("1. 🧪 Run module tests")
    print("2. 🎯 Run training examples")
    print("3. 📊 Run comprehensive evaluation")
    print("4. 📓 Open Jupyter notebook")
    print("5. 🚀 Run complete project (run.sh)")
    print()

    try:
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            print("\n🧪 Running module tests...")
            subprocess.run([sys.executable, "test_modules.py"], check=True)

        elif choice == "2":
            print("\n🎯 Running training examples...")
            subprocess.run([sys.executable, "training_examples.py"], check=True)

        elif choice == "3":
            print("\n📊 Running comprehensive evaluation...")
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
sys.path.append('.')
from utils import run_comprehensive_evaluation
results = run_comprehensive_evaluation()
print('✅ Evaluation completed!')
""",
                ]
            )

        elif choice == "4":
            print("\n📓 Starting Jupyter notebook...")
            subprocess.run(["jupyter", "notebook", "CA14.ipynb"], check=True)

        elif choice == "5":
            print("\n🚀 Running complete project...")
            subprocess.run(["./run.sh"], check=True)

        else:
            print("❌ Invalid choice. Please enter 1-5.")
            return 1

        print("\n✅ Operation completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⏹️ Operation cancelled by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

