#!/usr/bin/env python3
"""
Test script for CA19 Modular package

This script validates that the package can be imported and basic functionality works.
Run with: python test_package.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing package imports...")

    try:
        import ca19_modular

        print("✅ Main package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main package: {e}")
        return False

    # Test individual modules
    modules_to_test = [
        "ca19_modular.hybrid_quantum_classical_rl",
        "ca19_modular.neuromorphic_rl",
        "ca19_modular.quantum_rl",
        "ca19_modular.environments",
        "ca19_modular.experiments",
        "ca19_modular.utils",
    ]

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            return False

    return True


def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("\nTesting basic functionality...")

    try:
        from ca19_modular.utils import MissionConfig, create_default_config

        config = create_default_config()
        print(f"✅ MissionConfig created: state_dim={config.state_dim}")
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

    try:
        from ca19_modular.utils import get_available_modules, get_module_info

        modules = get_available_modules()
        print(f"✅ Available modules: {len(modules)} found")

        if modules:
            info = get_module_info(modules[0])
            print(f"✅ Module info retrieved for {modules[0]}")
    except Exception as e:
        print(f"❌ Failed to get module info: {e}")
        return False

    return True


def test_package_info():
    """Test package information functions"""
    print("\nTesting package information...")

    try:
        import ca19_modular

        ca19_modular.print_package_info()
        print("✅ Package info printed successfully")
    except Exception as e:
        print(f"❌ Failed to print package info: {e}")
        return False

    return True


def main():
    """Main test function"""
    print("🧪 Testing CA19 Modular Package")
    print("=" * 40)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False

    # Test package info
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
