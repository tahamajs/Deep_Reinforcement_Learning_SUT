#!/usr/bin/env python3
"""
Basic Test Script for CA8 Advanced Components
=============================================

This script tests the basic functionality of the advanced components
without requiring heavy dependencies.

Author: DRL Course Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")

    try:
        import numpy as np

        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ NumPy import failed: {e}")
        return False

    try:
        import matplotlib.pyplot as plt

        print("  ✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"  ❌ Matplotlib import failed: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔍 Testing basic functionality...")

    try:
        # Test NumPy
        data = np.random.randn(100, 4)
        print(f"  ✅ NumPy data generation: {data.shape}")

        # Test Matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(data[:, 0], data[:, 1], "o", alpha=0.6)
        ax.set_title("Test Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Save test plot
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/test_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("  ✅ Matplotlib plotting and saving successful")

        return True

    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False


def test_advanced_components():
    """Test advanced components (without heavy dependencies)"""
    print("\n🔍 Testing advanced components...")

    try:
        # Test if we can import our modules (without running them)
        from algorithms.advanced_causal_discovery import AdvancedCausalDiscovery

        print("  ✅ Advanced Causal Discovery imported successfully")

        from algorithms.advanced_multimodal_fusion import TransformerCrossModalAttention

        print("  ✅ Advanced Multi-Modal Fusion imported successfully")

        from algorithms.advanced_counterfactual_reasoning import StructuralCausalModel

        print("  ✅ Advanced Counterfactual Reasoning imported successfully")

        from algorithms.advanced_meta_transfer_learning import MAMLCausalLearner

        print("  ✅ Advanced Meta-Learning imported successfully")

        return True

    except Exception as e:
        print(f"  ❌ Advanced components test failed: {e}")
        return False


def create_test_visualization():
    """Create a test visualization"""
    print("\n🎨 Creating test visualization...")

    try:
        # Create a comprehensive test plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Random data
        data = np.random.randn(100, 4)
        axes[0, 0].scatter(data[:, 0], data[:, 1], c=data[:, 2], alpha=0.6)
        axes[0, 0].set_title("Test Scatter Plot")
        axes[0, 0].set_xlabel("Feature 1")
        axes[0, 0].set_ylabel("Feature 2")

        # Plot 2: Line plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)
        axes[0, 1].plot(x, y, linewidth=2)
        axes[0, 1].set_title("Test Line Plot")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Bar plot
        categories = ["Algorithm A", "Algorithm B", "Algorithm C", "Algorithm D"]
        values = [0.85, 0.78, 0.92, 0.88]
        bars = axes[1, 0].bar(
            categories,
            values,
            color=["skyblue", "lightgreen", "lightcoral", "lightyellow"],
        )
        axes[1, 0].set_title("Test Bar Plot")
        axes[1, 0].set_ylabel("Performance")
        axes[1, 0].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Plot 4: Heatmap
        heatmap_data = np.random.randn(8, 8)
        im = axes[1, 1].imshow(heatmap_data, cmap="viridis", aspect="equal")
        axes[1, 1].set_title("Test Heatmap")
        axes[1, 1].set_xlabel("X")
        axes[1, 1].set_ylabel("Y")
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(
            "visualizations/test_comprehensive_plot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("  ✅ Comprehensive test visualization created successfully")
        return True

    except Exception as e:
        print(f"  ❌ Test visualization creation failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 CA8 Advanced Components - Basic Test Suite")
    print("=" * 60)

    # Create necessary directories
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("✅ Directories created: visualizations/, results/, logs/")

    # Run tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Advanced Components", test_advanced_components),
        ("Test Visualization", create_test_visualization),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n📊 Running {test_name}...")
        print("-" * 40)

        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} passed!")
            else:
                print(f"❌ {test_name} failed!")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")

    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n🎉 All tests passed! CA8 Advanced Components are ready!")
        print("\n📁 Generated Files:")
        print("  - visualizations/test_plot.png")
        print("  - visualizations/test_comprehensive_plot.png")
        print("\n🚀 You can now run the full CA8 suite with:")
        print("  ./run.sh")
        print("  python3 main.py")
    else:
        print(
            f"\n⚠️  {total - passed} tests failed. Check the output above for details."
        )
        print("🔧 You may need to install missing dependencies:")
        print("  pip install numpy matplotlib")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


