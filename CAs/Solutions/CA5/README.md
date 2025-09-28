# CA5: Deep Q-Networks (DQN) and Advanced Value-Based Methods

## Overview

This assignment explores **Deep Q-Networks (DQN)** and advanced value-based reinforcement learning methods. The focus is on combining deep neural networks with Q-learning to handle high-dimensional state spaces, along with techniques to improve stability and sample efficiency.

## Key Concepts

### Deep Q-Networks (DQN)

- **Function Approximation**: Neural networks for Q-value estimation
- **Experience Replay**: Breaking temporal correlations in training data
- **Target Networks**: Stabilizing training with fixed Q-targets
- **Convolutional Architectures**: Handling image-based observations

### Advanced DQN Variants

- **Double DQN**: Addressing overestimation bias in Q-learning
- **Dueling DQN**: Separating state value and advantage estimation
- **Prioritized Experience Replay**: Intelligent sampling based on TD error
- **Rainbow DQN**: Combining multiple DQN improvements

## Project Structure

```
CA5/
├── CA5.ipynb                    # Main educational notebook
├── dqn_base.py                  # Basic DQN implementation
├── double_dqn.py                # Double DQN with bias correction
├── dueling_dqn.py               # Dueling architecture implementation
├── prioritized_replay.py        # Prioritized experience replay
├── rainbow_dqn.py               # Rainbow DQN combination
├── analysis_tools.py            # Performance analysis utilities
├── ca5_helpers.py               # Helper functions and utilities
├── ca5_main.py                  # Main training and evaluation script
├── __init__.py                  # Package initialization
└── requirements.txt             # Dependencies
```

## Algorithms Implemented

### 1. Basic DQN (`dqn_base.py`)

- **Core Components**:
  - Fully connected and convolutional neural networks
  - Experience replay buffer with uniform sampling
  - Target network updates for stability
  - ε-greedy exploration with decay
- **Key Features**:
  - Gradient clipping for training stability
  - Configurable network architectures
  - Comprehensive training metrics tracking

### 2. Double DQN (`double_dqn.py`)

- **Bias Correction**: Decouples action selection from evaluation
- **Implementation**: Uses online network for action selection, target network for evaluation
- **Analysis Tools**: Synthetic environments for bias demonstration
- **Performance Comparison**: Statistical comparison with standard DQN

### 3. Dueling DQN (`dueling_dqn.py`)

- **Architecture**: Separate value and advantage streams
- **Decomposition**: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
- **Benefits**: Better state value learning, improved sample efficiency
- **Variants**: Standard and convolutional dueling architectures

### 4. Prioritized Experience Replay (`prioritized_replay.py`)

- **Priority Calculation**: Based on TD error magnitude
- **Data Structure**: Sum tree for efficient sampling
- **Importance Sampling**: Bias correction weights
- **Hyperparameters**: Configurable prioritization strength

### 5. Rainbow DQN (`rainbow_dqn.py`)

- **Combined Improvements**: Integrates all advanced techniques
- **Multi-Head Networks**: Handles different aspects simultaneously
- **Performance**: State-of-the-art DQN performance
- **Modular Design**: Easy to enable/disable individual components

## Key Features

### Modular Architecture

- **Base Classes**: Reusable components for all DQN variants
- **Inheritance Hierarchy**: Easy extension and customization
- **Configuration System**: Hyperparameter management
- **Plugin Architecture**: Mix and match improvements

### Comprehensive Analysis Tools (`analysis_tools.py`)

- **Performance Metrics**: Learning curves, convergence analysis
- **Statistical Comparison**: Multiple runs with confidence intervals
- **Bias Analysis**: Overestimation bias visualization
- **Architecture Comparison**: Parameter efficiency analysis

### Training Framework (`ca5_main.py`)

- **Experiment Runner**: Automated training and evaluation
- **Hyperparameter Sweeps**: Systematic parameter optimization
- **Benchmark Suite**: Multi-environment testing
- **Result Visualization**: Comprehensive plotting utilities

## Installation & Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: Neural network implementation and optimization
- **Gym/Gymnasium**: Reinforcement learning environments
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization and plotting
- **OpenCV**: Image processing for Atari environments

## Usage Examples

### Basic DQN Training

```python
from dqn_base import DQNAgent, create_test_environment

# Create environment and agent
env, state_size, action_size = create_test_environment()
agent = DQNAgent(state_size, action_size)

# Train agent
scores, losses = agent.train(env, num_episodes=1000)
print(f"Final score: {scores[-1]:.2f}")
```

### Double DQN with Bias Analysis

```python
from double_dqn import DoubleDQNAgent, OverestimationAnalysis

# Train Double DQN agent
agent = DoubleDQNAgent(state_size, action_size)
scores, _ = agent.train(env, num_episodes=500)

# Analyze overestimation bias
bias_analysis = OverestimationAnalysis()
results = bias_analysis.visualize_bias_analysis()
```

### Algorithm Comparison

```python
from analysis_tools import DQNComparison

# Compare multiple algorithms
comparison = DQNComparison(env, state_size, action_size)
standard_results, double_results, _, _ = comparison.run_comparison()
comparison.visualize_comparison(standard_results, double_results)
```

## Educational Content

### CA5.ipynb Features

- **DQN Fundamentals**: From tabular Q-learning to deep networks
- **Experience Replay**: Understanding temporal correlation issues
- **Target Networks**: Mathematical analysis of training stability
- **Double DQN**: Overestimation bias theory and correction
- **Dueling Architecture**: Value-advantage decomposition
- **Prioritized Replay**: Intelligent sampling strategies
- **Rainbow Integration**: Combining all improvements
- **Atari Gaming**: Application to complex visual environments

### Key Learning Objectives

1. **Deep Function Approximation**: Neural networks for value functions
2. **Training Stability**: Target networks and gradient clipping
3. **Sample Efficiency**: Experience replay and prioritization
4. **Bias Correction**: Understanding and fixing overestimation
5. **Architecture Design**: Dueling networks and multi-head architectures
6. **Hyperparameter Tuning**: Systematic optimization strategies

## Performance & Results

### Expected Performance (CartPole-v1)

- **Basic DQN**: ~150-200 average reward
- **Double DQN**: ~180-250 average reward
- **Dueling DQN**: ~200-280 average reward
- **Rainbow DQN**: ~250-350 average reward

### Training Stability Improvements

- **Experience Replay**: Reduces variance by ~50%
- **Target Networks**: Prevents divergence and oscillations
- **Double DQN**: Reduces overestimation bias by ~20-30%
- **Prioritized Replay**: Improves sample efficiency by ~30-50%

## Advanced Topics

### Convolutional Architectures

- **Atari Preprocessing**: Frame stacking and downsampling
- **Network Design**: Convolutional layers for feature extraction
- **Training Challenges**: High-dimensional input handling

### Hyperparameter Sensitivity

- **Learning Rate**: Critical for stability (typically 1e-4 to 1e-3)
- **Target Update Frequency**: Trade-off between adaptation and stability
- **Replay Buffer Size**: Memory vs sample diversity balance
- **Prioritization Strength**: α parameter tuning

### Extensions and Variations

- **Distributional DQN**: Learning value distributions
- **Noisy Networks**: Intrinsic exploration mechanisms
- **Multi-Step Learning**: N-step returns for bootstrapping
- **Population-Based Training**: Automated hyperparameter optimization

## Applications

### Game Playing

- **Atari Games**: Classic arcade game playing
- **Board Games**: Go, Chess, and other strategy games
- **Real-Time Strategy**: Complex multi-agent scenarios

### Robotics

- **Continuous Control**: Motor control and manipulation
- **Navigation**: Autonomous path planning
- **Manipulation**: Object grasping and tool use

### Resource Management

- **Network Routing**: Traffic optimization
- **Power Systems**: Load balancing and scheduling
- **Financial Trading**: Portfolio optimization

## Troubleshooting

### Common Issues

- **Unstable Training**: Use target networks and gradient clipping
- **Poor Sample Efficiency**: Implement prioritized replay
- **Overestimation**: Switch to Double DQN
- **Slow Convergence**: Adjust learning rates and network architecture

### Performance Tips

- **Batch Normalization**: Add to convolutional layers for stability
- **Learning Rate Scheduling**: Decay learning rate over time
- **Data Augmentation**: Artificially increase replay diversity
- **Multi-Environment Training**: Transfer learning across tasks

### Debugging Tools

- **Q-Value Monitoring**: Track average Q-values during training
- **Gradient Analysis**: Monitor gradient magnitudes and distributions
- **Replay Buffer Inspection**: Analyze stored experience distributions
- **Network Output Visualization**: Understand what the network has learned

---

_This assignment provides a comprehensive exploration of deep Q-learning, from basic DQN to state-of-the-art Rainbow DQN, with practical implementations and thorough analysis tools for understanding value-based reinforcement learning._
