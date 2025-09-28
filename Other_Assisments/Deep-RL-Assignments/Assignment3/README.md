# Assignment 3: Asynchronous Advantage Actor-Critic (A3C)

Author: Taha Majlesi - 810101504, University of Tehran

This assignment implements the Asynchronous Advantage Actor-Critic (A3C) algorithm for reinforcement learning on Atari games.

## Structure

- `src/preprocessing.py`: Frame preprocessing utilities for Atari environments
- `src/model.py`: Actor-Critic neural network architecture
- `src/a3c_agent.py`: A3C agent implementation with training and testing
- `run_a3c.py`: Main script to run A3C training
- `requirements.txt`: Python dependencies

## Algorithm

**A3C (Asynchronous Advantage Actor-Critic)** is a policy gradient method that uses multiple workers to asynchronously update a global network. Key features:

- **Actor-Critic Architecture**: Separate policy (actor) and value (critic) networks
- **Advantage Function**: Uses advantage estimates for reduced variance
- **Asynchronous Updates**: Multiple workers update a global network asynchronously
- **N-step Returns**: Uses n-step returns for bootstrapping

## Environment

- **Breakout-v0**: Atari Breakout game from Gymnasium
- Frame preprocessing: 84x84 grayscale frames
- Reward normalization and stacking

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python run_a3c.py --num-episodes 50000 --policy_lr 5e-4 --critic_lr 1e-4
```

### Testing with Rendering

```bash
python run_a3c.py --render --weights_path path/to/model.h5
```

### Custom Parameters

```bash
python run_a3c.py \
    --num-episodes 100000 \
    --policy_lr 1e-4 \
    --critic_lr 5e-5 \
    --n 50 \
    --test_episodes 50 \
    --save_interval 5000
```

## Command Line Arguments

- `--num-episodes`: Number of training episodes (default: 50000)
- `--policy_lr`: Actor learning rate (default: 5e-4)
- `--critic_lr`: Critic learning rate (default: 1e-4)
- `--n`: N-step returns parameter (default: 100)
- `--reward_norm`: Reward normalization factor (default: 100.0)
- `--random_seed`: Random seed for reproducibility (default: 999)
- `--test_episodes`: Number of test episodes (default: 25)
- `--save_interval`: Model save interval (default: 1000)
- `--test_interval`: Testing interval (default: 250)
- `--log_interval`: Logging interval (default: 25)
- `--weights_path`: Path to pretrained weights
- `--det_eval`: Use deterministic policy for testing
- `--render`: Render environment during testing

## Features

- **Multi-worker Training**: Asynchronous updates from multiple workers
- **Experience Replay**: N-step returns for improved sample efficiency
- **TensorBoard Logging**: Real-time monitoring of training metrics
- **Model Checkpointing**: Automatic saving of model weights
- **Video Recording**: Save gameplay videos during testing
- **Performance Plotting**: Generate reward curves with error bars

## Dependencies

- gymnasium: Modern RL environments
- torch: Deep learning framework
- torchvision: Computer vision utilities
- numpy: Numerical computations
- opencv-python: Image processing
- tensorboardX: Logging and visualization
- matplotlib: Plotting

## Results

The A3C algorithm learns to play Atari Breakout by maximizing the score through paddle and ball interactions. Training involves optimizing both the policy and value networks using asynchronous updates from multiple workers.
