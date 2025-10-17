# Random Network Distillation (RND) with PPO - Implementation Guide

## Overview

This project implements Random Network Distillation (RND) combined with Proximal Policy Optimization (PPO) for exploration in reinforcement learning. The implementation is designed to work with the MiniGrid environment.

## What Was Implemented

### 1. TargetModel (`Brain/model.py`)

- **Architecture**: 3 convolutional layers (32→64→128 channels) + 1 fully connected layer (512 features)
- **Purpose**: Fixed random network that generates target features for the predictor to learn
- **Initialization**: Orthogonal weight initialization with gain=√2
- **Forward Pass**: Normalizes input, applies convolutions with ReLU, flattens, and outputs 512-dim features

### 2. PredictorModel (`Brain/model.py`)

- **Architecture**: Same conv layers as TargetModel + 2 additional FC layers (512→512→512)
- **Purpose**: Trainable network that learns to predict the target model's features
- **Initialization**: Orthogonal weights with gain=√2 for hidden layers, gain=√0.01 for final layer
- **Forward Pass**: Same preprocessing as TargetModel, plus additional FC layers

### 3. Intrinsic Reward Calculation (`Brain/brain.py`)

- **Method**: `calculate_int_rewards()`
- **Process**:
  1. Normalize observations using running statistics
  2. Extract features from both target and predictor models
  3. Compute mean squared error between features
  4. Return prediction error as intrinsic reward

### 4. RND Loss Calculation (`Brain/brain.py`)

- **Method**: `calculate_rnd_loss()`
- **Process**:
  1. Compute squared error between predictor and target features
  2. Apply dropout mask using `predictor_proportion` (0.25 by default)
  3. Return mean loss for training the predictor

## Key Implementation Details

### Architecture Design

- **Convolutional Layers**: 3x3 kernels with padding=1 to preserve spatial dimensions
- **Feature Dimension**: 512-dimensional feature vectors for both models
- **Activation**: ReLU activations after each conv layer

### Weight Initialization

- **Orthogonal Initialization**: Used for all conv and linear layers
- **Gain Values**: √2 for most layers, √0.01 for final predictor layer (slower learning)

### Normalization Strategy

- **Input Normalization**: Observations normalized to [0,1] range by dividing by 255
- **Running Statistics**: Used for observation normalization in intrinsic reward calculation
- **Clipping**: Values clipped to [-5, 5] range after normalization

### Training Strategy

- **Dropout Mask**: Only 25% of features trained per batch (configurable via `predictor_proportion`)
- **Fixed Target**: Target model parameters frozen (no gradients)
- **Combined Loss**: RND loss added to PPO loss for joint training

## File Structure

```
HW10_PPO-RND/
├── Brain/
│   ├── brain.py          # Core agent logic and training
│   └── model.py          # Neural network architectures
├── Common/
│   ├── config.py         # Hyperparameters
│   ├── utils.py          # Utility functions
│   ├── logger.py         # TensorBoard logging
│   └── runner.py         # Environment runner
├── main.py               # Main training script
├── test_implementation.py # Test script
├── requirements.txt      # Dependencies
└── HW10_RND.ipynb       # Jupyter notebook
```

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Implementation

```bash
python test_implementation.py
```

### 3. Train the Agent

```bash
python main.py --train_from_scratch
```

### 4. Evaluate Trained Agent

```bash
python main.py --do_test
```

## Configuration

Key hyperparameters in `config.py`:

- `predictor_proportion`: Fraction of features to train per batch (0.25)
- `int_gamma`: Discount factor for intrinsic rewards (0.99)
- `ext_gamma`: Discount factor for extrinsic rewards (0.99)
- `lr`: Learning rate (2.5e-4)
- `n_epochs`: PPO training epochs (4)

## Expected Behavior

1. **Exploration**: Agent should explore more in unseen areas due to high intrinsic rewards
2. **Learning**: Predictor should gradually learn to match target features in visited areas
3. **Performance**: Should achieve better exploration compared to standard PPO

## Troubleshooting

### Common Issues:

1. **CUDA errors**: Ensure PyTorch CUDA version matches your system
2. **Memory issues**: Reduce batch size or rollout length
3. **Slow training**: Check if GPU is being used (`torch.cuda.is_available()`)

### Debugging:

- Use `test_implementation.py` to verify model architectures
- Check TensorBoard logs for training progress
- Monitor intrinsic vs extrinsic reward ratios

## References

- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [MiniGrid Environment](https://github.com/maximecb/gym-minigrid)
