#!/usr/bin/env python
"""
Quick test script to verify environment setup

This script checks if all required packages are installed and working.
"""

import sys

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking required packages...\n")
    
    packages = {
        'tensorflow': 'TensorFlow',
        'gym': 'OpenAI Gym',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - NOT FOUND")
            all_ok = False
    
    # Check MuJoCo separately
    print("\nChecking MuJoCo...")
    try:
        import mujoco_py
        print(f"✓ MuJoCo-py          - OK")
        print(f"  Version: {mujoco_py.__version__}")
    except ImportError:
        print(f"✗ MuJoCo-py          - NOT FOUND")
        all_ok = False
    except Exception as e:
        print(f"⚠ MuJoCo-py          - ERROR: {e}")
        all_ok = False
    
    return all_ok

def check_expert_policies():
    """Check if expert policy files exist."""
    import os
    
    print("\n" + "="*60)
    print("Checking expert policies...\n")
    
    environments = [
        "Hopper-v2", "Ant-v2", "HalfCheetah-v2",
        "Walker2d-v2", "Reacher-v2", "Humanoid-v2"
    ]
    
    all_ok = True
    for env in environments:
        policy_file = f"experts/{env}.pkl"
        if os.path.exists(policy_file):
            size = os.path.getsize(policy_file) / 1024  # KB
            print(f"✓ {env:20s} - {size:.1f} KB")
        else:
            print(f"✗ {env:20s} - NOT FOUND")
            all_ok = False
    
    return all_ok

def test_gym_environment():
    """Test if a simple Gym environment can be created."""
    print("\n" + "="*60)
    print("Testing Gym environment creation...\n")
    
    try:
        import gym
        env = gym.make('Hopper-v2')
        obs = env.reset()
        print(f"✓ Successfully created Hopper-v2 environment")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation shape: {obs.shape}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False

def main():
    print("="*60)
    print("BEHAVIORAL CLONING ENVIRONMENT CHECK")
    print("="*60)
    print()
    
    # Check imports
    imports_ok = check_imports()
    
    # Check expert policies
    policies_ok = check_expert_policies()
    
    # Test gym
    gym_ok = test_gym_environment()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if imports_ok and policies_ok and gym_ok:
        print("✓ All checks passed! You're ready to run the pipeline.")
        print("\nTo run the pipeline, execute:")
        print("  python run_full_pipeline.py --env Hopper-v2 --num_rollouts 10 --epochs 50")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        if not imports_ok:
            print("\nTo install missing packages:")
            print("  pip install -r requirements.txt")
        if not gym_ok:
            print("\nMuJoCo may not be properly installed.")
            print("See RUNNING_INSTRUCTIONS.md for setup help.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
