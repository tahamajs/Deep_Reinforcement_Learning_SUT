# DDPG, TD3, and HER Implementation

**Author:** Saeed Reza Zouashkiani (Student ID: 400206262)

This repository contains a modular implementation of Deep Deterministic Policy Gradient (DDPG), Twin Delayed Deep Deterministic Policy Gradient (TD3), and Hindsight Experience Replay (HER) algorithms for continuous control tasks.

## Overview

The implementation is organized into modular components for better maintainability and reusability:

- **Actor Network**: Policy network for action selection
- **Critic Network**: Value function network(s) for Q-value estimation
- **Replay Buffer**: Experience replay buffer for storing and sampling transitions
- **DDPG Agent**: Main agent class orchestrating the training process

## Project Structure

```
Assignment4/
├── src/
│   ├── __init__.py
│   ├── actor.py              # ActorNetwork class
│   ├── critic.py             # CriticNetwork and CriticNetworkTD3 classes
│   ├── replay_buffer.py      # ReplayBuffer class
│   └── ddpg_agent.py         # DDPGAgent main class
├── run_ddpg.py               # Main training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── algo/                    # Original monolithic implementation (deprecated)
    └── ...
```

## Features

### Algorithms Supported

- **DDPG**: Deep Deterministic Policy Gradient
- **TD3**: Twin Delayed DDPG (with twin critics and delayed policy updates)
- **HER**: Hindsight Experience Replay (relabels failed trajectories with achieved goals)

### Key Components

- Modular neural network architectures
- Experience replay with prioritized sampling support
- Action noise for exploration
- Soft target network updates
- Comprehensive logging and visualization
- Model checkpointing and loading

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have the custom environments available (the `envs` module should be in your Python path).

## Usage

### Training

Run the main training script with default parameters:

```bash
python run_ddpg.py
```

### Custom Parameters

You can customize various hyperparameters:

```bash
python run_ddpg.py \
    --algorithm ddpg \
    --num-episodes 50000 \
    --actor-lr 1e-4 \
    --critic-lr 1e-3 \
    --batch-size 1024 \
    --gamma 0.98 \
    --render
```

### Available Arguments

- `--algorithm`: Choose algorithm (`ddpg`, `td3`, `her`)
- `--num-episodes`: Number of training episodes
- `--actor-lr`: Actor network learning rate
- `--critic-lr`: Critic network learning rate
- `--batch-size`: Training batch size
- `--gamma`: Discount factor
- `--buffer-size`: Replay buffer size
- `--epsilon`: Exploration noise parameter
- `--render`: Enable environment rendering during evaluation

## Environment

The implementation is designed to work with the `Pushing2D-v0` environment, which involves:

- A pusher agent that must move a puck to a goal location
- Continuous action space (2D forces)
- Sparse rewards with goal-based success detection
- Support for HER relabeling

## Results

The training process generates:

- **TensorBoard logs**: Training metrics and losses
- **Model checkpoints**: Saved model weights at regular intervals
- **Evaluation plots**: Success rates and reward curves
- **Trajectory visualizations**: Sample episode trajectories (when rendering is enabled)

## Key Implementation Details

### Actor Network

- 3-layer MLP with ReLU activations
- Tanh output activation for bounded actions
- Target network for stable learning

### Critic Network

- State-action value function approximation
- TD3 uses twin critics for reduced overestimation bias
- Target networks with soft updates

### Training Loop

- Experience collection with exploration noise
- Batch sampling from replay buffer
- Alternating actor and critic updates
- HER trajectory relabeling (when enabled)

### HER Implementation

- Relabels failed trajectories with achieved goals
- Sparse reward environment support
- Automatic goal detection and reward calculation

## References

- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [HER Paper](https://arxiv.org/abs/1707.01495)

## Notes

- The original monolithic implementation is preserved in the `algo/` directory for reference
- The modular structure improves code maintainability and testing
- All components are designed to be easily extensible for other continuous control algorithms
