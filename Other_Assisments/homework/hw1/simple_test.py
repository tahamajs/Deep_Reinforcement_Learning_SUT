#!/usr/bin/env python
"""
Simple Test - Run a Quick Behavioral Cloning Demo

This script runs a minimal test of the behavioral cloning pipeline
using just 5 rollouts and 20 epochs for quick verification.

Usage: python simple_test.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SIMPLE BEHAVIORAL CLONING TEST")
print("="*70)
print("\nThis is a quick test with minimal parameters:")
print("  - Environment: Hopper-v2 (fastest)")
print("  - Rollouts: 5 (instead of 20)")
print("  - Epochs: 20 (instead of 100)")
print("  - Expected runtime: 2-3 minutes")
print("\n" + "="*70 + "\n")

# Check if we can import required modules
try:
    print("Checking imports...")
    import tensorflow as tf
    
    # TensorFlow 2.x compatibility
    if hasattr(tf, '__version__') and int(tf.__version__.split('.')[0]) >= 2:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
    
    import gym
    import numpy as np
    print("✓ All required modules found\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Check if expert policy exists
expert_policy = "experts/Hopper-v2.pkl"
if not os.path.exists(expert_policy):
    print(f"✗ Expert policy not found: {expert_policy}")
    sys.exit(1)

print("✓ Expert policy found\n")
print("Starting pipeline...\n")
print("="*70 + "\n")

# Import and run the pipeline
try:
    from src.expert_data_collector import ExpertDataCollector, load_expert_policy
    from src.behavioral_cloning import BehavioralCloning
    import pickle
    
    # Parameters for quick test
    ENV_NAME = "Hopper-v2"
    NUM_ROLLOUTS = 5
    EPOCHS = 20
    BATCH_SIZE = 32
    
    # Step 1: Collect expert data
    print("STEP 1/3: Collecting expert data...")
    with tf.Session() as sess:
        policy_fn = load_expert_policy(expert_policy)
        collector = ExpertDataCollector(policy_fn, ENV_NAME)
        expert_data = collector.collect_rollouts(NUM_ROLLOUTS, render=False)
        
        os.makedirs("expert_data", exist_ok=True)
        data_file = f"expert_data/{ENV_NAME}_test.pkl"
        collector.save_data(expert_data, data_file)
    
    print(f"\n✓ Expert mean return: {expert_data['mean_return']:.2f}\n")
    
    # Step 2: Train BC policy
    print("STEP 2/3: Training BC policy...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    observations = data["observations"]
    actions = data["actions"]
    
    env = gym.make(ENV_NAME)
    bc_agent = BehavioralCloning(env, learning_rate=1e-3, hidden_sizes=[100, 100])
    history = bc_agent.train(observations, actions, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print(f"\n✓ Training complete. Final loss: {history['losses'][-1]:.6f}\n")
    
    # Step 3: Quick evaluation
    print("STEP 3/3: Evaluating policy (3 episodes)...")
    results = bc_agent.evaluate(num_episodes=3, render=False)
    
    bc_agent.sess.close()
    
    # Summary
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nExpert Performance:  {expert_data['mean_return']:.2f}")
    print(f"BC Policy Performance: {results['mean_return']:.2f}")
    print(f"Performance Ratio:     {results['mean_return']/expert_data['mean_return']*100:.1f}%")
    print("\n" + "="*70)
    print("\n✓ Everything works! Now you can run the full pipeline:")
    print("  ./run_pipeline.sh")
    print("\nOr:")
    print(f"  python run_full_pipeline.py --env Hopper-v2 --num_rollouts 20 --epochs 100")
    print("\n" + "="*70)
    
except Exception as e:
    print(f"\n✗ Error during execution: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
    print("2. Check if MuJoCo is properly installed")
    print("3. See RUNNING_INSTRUCTIONS.md for detailed setup help")
    sys.exit(1)
