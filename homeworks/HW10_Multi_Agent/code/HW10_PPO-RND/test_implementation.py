#!/usr/bin/env python3
"""
Test script to verify the RND implementation works correctly.
This script tests the model architectures and basic functionality.
"""

import torch
import numpy as np
from Brain.model import PolicyModel, TargetModel, PredictorModel
from Brain.brain import Brain


def test_models():
    """Test the model architectures."""
    print("Testing model architectures...")

    # Test parameters
    state_shape = (3, 7, 7)  # MiniGrid observation shape
    n_actions = 4
    batch_size = 2

    # Create models
    policy_model = PolicyModel(state_shape, n_actions)
    target_model = TargetModel(state_shape)
    predictor_model = PredictorModel(state_shape)

    # Create dummy input
    dummy_input = torch.randn(batch_size, *state_shape)
    dummy_hidden = torch.zeros(batch_size, 256)

    print(f"Input shape: {dummy_input.shape}")

    # Test PolicyModel
    try:
        dist, int_val, ext_val, probs, hidden = policy_model(dummy_input, dummy_hidden)
        print(f"✓ PolicyModel forward pass successful")
        print(f"  - Distribution: {dist}")
        print(f"  - Intrinsic value shape: {int_val.shape}")
        print(f"  - Extrinsic value shape: {ext_val.shape}")
        print(f"  - Hidden state shape: {hidden.shape}")
    except Exception as e:
        print(f"✗ PolicyModel failed: {e}")
        return False

    # Test TargetModel
    try:
        target_features = target_model(dummy_input)
        print(f"✓ TargetModel forward pass successful")
        print(f"  - Target features shape: {target_features.shape}")
        assert target_features.shape == (
            batch_size,
            512,
        ), f"Expected (batch_size, 512), got {target_features.shape}"
    except Exception as e:
        print(f"✗ TargetModel failed: {e}")
        return False

    # Test PredictorModel
    try:
        pred_features = predictor_model(dummy_input)
        print(f"✓ PredictorModel forward pass successful")
        print(f"  - Predicted features shape: {pred_features.shape}")
        assert pred_features.shape == (
            batch_size,
            512,
        ), f"Expected (batch_size, 512), got {pred_features.shape}"
    except Exception as e:
        print(f"✗ PredictorModel failed: {e}")
        return False

    return True


def test_brain_functionality():
    """Test the Brain class functionality."""
    print("\nTesting Brain functionality...")

    # Create config
    config = {
        "state_shape": (3, 7, 7),
        "obs_shape": (3, 7, 7),
        "n_actions": 4,
        "lr": 2.5e-4,
        "int_gamma": 0.99,
        "ext_gamma": 0.99,
        "lambda": 0.95,
        "clip_range": 0.1,
        "max_grad_norm": 0.5,
        "ent_coeff": 0.001,
        "ext_adv_coeff": 1.0,
        "int_adv_coeff": 1.0,
        "predictor_proportion": 0.25,
        "n_epochs": 1,
    }

    try:
        brain = Brain(**config)
        print("✓ Brain initialization successful")
    except Exception as e:
        print(f"✗ Brain initialization failed: {e}")
        return False

    # Test intrinsic reward calculation
    try:
        dummy_obs = np.random.randint(0, 255, (3, 7, 7), dtype=np.uint8)
        int_reward = brain.calculate_int_rewards(dummy_obs, batch=False)
        print(f"✓ Intrinsic reward calculation successful")
        print(f"  - Intrinsic reward shape: {int_reward.shape}")
        print(f"  - Intrinsic reward value: {int_reward[0]:.6f}")
    except Exception as e:
        print(f"✗ Intrinsic reward calculation failed: {e}")
        return False

    # Test RND loss calculation
    try:
        dummy_obs_tensor = torch.randn(2, 3, 7, 7)
        rnd_loss = brain.calculate_rnd_loss(dummy_obs_tensor)
        print(f"✓ RND loss calculation successful")
        print(f"  - RND loss value: {rnd_loss.item():.6f}")
    except Exception as e:
        print(f"✗ RND loss calculation failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("RND Implementation Test")
    print("=" * 50)

    # Test models
    models_ok = test_models()

    # Test brain functionality
    brain_ok = test_brain_functionality()

    print("\n" + "=" * 50)
    if models_ok and brain_ok:
        print("✓ All tests passed! Implementation is working correctly.")
        print("You can now proceed to train the agent.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    print("=" * 50)


if __name__ == "__main__":
    main()
