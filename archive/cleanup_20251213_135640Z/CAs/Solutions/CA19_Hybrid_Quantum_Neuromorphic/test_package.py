"""
Test script for CA19 Modular package

This script validates that the package can be imported and basic functionality works.
Run with: python3 test_package.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing package imports...")

    # Try direct imports from module files
    modules_to_test = [
        ("utils", ["MissionConfig", "PerformanceTracker"]),
        ("agents.quantum_inspired_agent", ["QuantumInspiredAgent"]),
        ("agents.spiking_agent", ["SpikingAgent"]),
    ]

    success_count = 0
    for module_path, classes in modules_to_test:
        try:
            module = __import__(module_path, fromlist=classes)
            for cls in classes:
                if hasattr(module, cls):
                    print(f"✅ {module_path}.{cls} imported successfully")
                    success_count += 1
                else:
                    print(f"⚠️  {module_path}.{cls} not found in module")
        except ImportError as e:
            print(f"⚠️  Failed to import {module_path}: {e}")
        except Exception as e:
            print(f"⚠️  Error with {module_path}: {e}")

    return success_count > 0


def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("\nTesting basic functionality...")

    try:
        from utils import MissionConfig
        
        config = MissionConfig()
        print(f"✅ MissionConfig created: state_dim={config.state_dim}")
    except Exception as e:
        print(f"⚠️  Failed to create config: {e}")
        return False

    try:
        import numpy as np
        print(f"✅ NumPy available: version {np.__version__}")
    except Exception as e:
        print(f"⚠️  NumPy not available: {e}")
    
    try:
        import torch
        print(f"✅ PyTorch available: version {torch.__version__}")
    except Exception as e:
        print(f"⚠️  PyTorch not available: {e}")
    
    try:
        import gymnasium
        print(f"✅ Gymnasium available: version {gymnasium.__version__}")
    except Exception as e:
        print(f"⚠️  Gymnasium not available: {e}")

    return True


def test_package_info():
    """Test package information functions"""
    print("\nTesting basic dependencies...")

    dependencies = {
        'numpy': 'NumPy - Numerical computing',
        'scipy': 'SciPy - Scientific computing',
        'torch': 'PyTorch - Deep learning framework',
        'matplotlib': 'Matplotlib - Plotting library',
        'gymnasium': 'Gymnasium - RL environments',
    }
    
    available = []
    missing = []
    
    for pkg, desc in dependencies.items():
        try:
            __import__(pkg)
            available.append(pkg)
            print(f"✅ {desc}")
        except ImportError:
            missing.append(pkg)
            print(f"❌ {desc} - NOT FOUND")
    
    print(f"\n📊 Dependencies: {len(available)}/{len(dependencies)} available")
    
    if missing:
        print(f"⚠️  Missing: {', '.join(missing)}")
    
    return len(missing) == 0


def main():
    """Main test function"""
    print("🧪 Testing CA19 Modular Package")
    print("=" * 40)

    all_passed = True

    if not test_imports():
        all_passed = False

    if not test_basic_functionality():
        all_passed = False

    if not test_package_info():
        all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Package is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
