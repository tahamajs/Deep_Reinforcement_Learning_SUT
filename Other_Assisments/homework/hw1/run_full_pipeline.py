#!/usr/bin/env python
"""
Complete Behavioral Cloning Pipeline Runner

This script runs the full behavioral cloning pipeline:
1. Collects expert data
2. Trains BC policy
3. Evaluates trained policy

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import os
import sys
import argparse
import pickle
import tensorflow as tf

# TensorFlow 2.x compatibility
if hasattr(tf, '__version__') and int(tf.__version__.split('.')[0]) >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import gym
import numpy as np
from src.expert_data_collector import ExpertDataCollector, load_expert_policy
from src.behavioral_cloning import BehavioralCloning


def run_full_pipeline(env_name, expert_policy_path, num_rollouts=20, epochs=100, 
                      batch_size=64, render_collect=False, render_eval=False):
    """Run the complete behavioral cloning pipeline."""
    
    print("="*80)
    print("BEHAVIORAL CLONING PIPELINE")
    print("="*80)
    print(f"Environment: {env_name}")
    print(f"Expert Policy: {expert_policy_path}")
    print(f"Number of Rollouts: {num_rollouts}")
    print(f"Training Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print("="*80)
    
    # Step 1: Collect Expert Data
    print("\n" + "="*80)
    print("STEP 1: COLLECTING EXPERT DATA")
    print("="*80)
    
    with tf.Session() as sess:
        print("Loading expert policy...")
        policy_fn = load_expert_policy(expert_policy_path)
        print("Expert policy loaded successfully!")
        
        print(f"\nCollecting {num_rollouts} expert rollouts...")
        collector = ExpertDataCollector(policy_fn, env_name)
        expert_data = collector.collect_rollouts(num_rollouts, render=render_collect)
        
        # Save expert data
        os.makedirs("expert_data", exist_ok=True)
        data_file = os.path.join("expert_data", f"{env_name}.pkl")
        collector.save_data(expert_data, data_file)
        
        print(f"\n✓ Expert data collection completed!")
        print(f"  - Mean return: {expert_data['mean_return']:.2f}")
        print(f"  - Std return: {expert_data['std_return']:.2f}")
        print(f"  - Total samples: {len(expert_data['observations'])}")
    
    # Step 2: Train Behavioral Cloning Policy
    print("\n" + "="*80)
    print("STEP 2: TRAINING BEHAVIORAL CLONING POLICY")
    print("="*80)
    
    print(f"Loading expert data from {data_file}...")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    observations = data["observations"]
    actions = data["actions"]
    
    print(f"Loaded {len(observations)} expert transitions")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Create environment and BC agent
    env = gym.make(env_name)
    print(f"\nEnvironment info:")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    print(f"\nTraining behavioral cloning policy...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: 1e-3")
    print(f"  - Hidden layers: [100, 100]")
    
    bc_agent = BehavioralCloning(env, learning_rate=1e-3, hidden_sizes=[100, 100])
    history = bc_agent.train(observations, actions, epochs=epochs, batch_size=batch_size)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_file = os.path.join("models", f"bc_policy_{env_name}")
    saver = tf.train.Saver()
    saver.save(bc_agent.sess, model_file)
    print(f"\n✓ Model saved to {model_file}")
    
    # Save training history
    history_file = os.path.join("models", f"training_history_{env_name}.pkl")
    with open(history_file, "wb") as f:
        pickle.dump(history, f)
    print(f"✓ Training history saved to {history_file}")
    
    print(f"\n✓ Training completed!")
    print(f"  - Final loss: {history['losses'][-1]:.6f}")
    
    # Step 3: Evaluate Trained Policy
    print("\n" + "="*80)
    print("STEP 3: EVALUATING TRAINED POLICY")
    print("="*80)
    
    print("Evaluating policy for 10 episodes...")
    results = bc_agent.evaluate(num_episodes=10, render=render_eval)
    
    print(f"\n✓ Evaluation completed!")
    print(f"  - Mean return: {results['mean_return']:.2f}")
    print(f"  - Std return: {results['std_return']:.2f}")
    
    # Compare with expert
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Expert Performance:")
    print(f"  - Mean return: {expert_data['mean_return']:.2f} ± {expert_data['std_return']:.2f}")
    print(f"\nBC Policy Performance:")
    print(f"  - Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"\nPerformance Ratio: {results['mean_return'] / expert_data['mean_return'] * 100:.1f}%")
    print("="*80)
    
    bc_agent.sess.close()
    
    return {
        'expert_data': expert_data,
        'training_history': history,
        'evaluation_results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Run complete behavioral cloning pipeline")
    parser.add_argument("--env", type=str, default="Hopper-v2", 
                       help="Gym environment name")
    parser.add_argument("--expert_policy", type=str, 
                       help="Path to expert policy file (default: experts/<env>.pkl)")
    parser.add_argument("--num_rollouts", type=int, default=20,
                       help="Number of expert rollouts to collect")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Training batch size")
    parser.add_argument("--render_collect", action="store_true",
                       help="Render environment during data collection")
    parser.add_argument("--render_eval", action="store_true",
                       help="Render environment during evaluation")
    
    args = parser.parse_args()
    
    # Set default expert policy path if not provided
    if args.expert_policy is None:
        args.expert_policy = os.path.join("experts", f"{args.env}.pkl")
    
    # Check if expert policy exists
    if not os.path.exists(args.expert_policy):
        print(f"Error: Expert policy file not found: {args.expert_policy}")
        sys.exit(1)
    
    # Run the pipeline
    results = run_full_pipeline(
        env_name=args.env,
        expert_policy_path=args.expert_policy,
        num_rollouts=args.num_rollouts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        render_collect=args.render_collect,
        render_eval=args.render_eval
    )
    
    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
