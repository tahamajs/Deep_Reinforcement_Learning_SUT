# CA4: Policy Gradient Methods and Neural Networks in RL

## Overview

This assignment explores **Policy Gradient Methods** and their implementation using **Neural Networks** in reinforcement learning. The focus is on direct policy optimization through gradient ascent on the expected return, covering both discrete and continuous action spaces.

## Key Concepts

### Policy Gradient Methods

- **REINFORCE Algorithm**: Monte Carlo policy gradient with optional baseline
- **Actor-Critic Methods**: Combining policy and value function approximations
- **Continuous Action Spaces**: Gaussian policies for continuous control
- **Neural Network Policies**: Parameterized policies using deep networks

### Mathematical Foundations

- **Score Function**: ∇_θ log π(a|s,θ) for policy gradient computation
- **Policy Gradient Theorem**: ∇*θ J(θ) = E[∇*θ log π(a|s,θ) \* Q(s,a)]
- **Baseline Subtraction**: Variance reduction using state value functions
- **Advantage Functions**: A(s,a) = Q(s,a) - V(s) for improved gradients

## Project Structure

```
CA4/
├── CA4_modular.ipynb          # Main educational notebook
├── algorithms.py              # Policy gradient implementations
├── policies.py                # Neural network policy architectures
├── environments.py            # Environment wrappers and utilities
├── experiments.py             # Experiment runners and benchmarks
├── visualization.py           # Policy and training visualization
├── exploration.py             # Exploration strategies (if present)
├── __init__.py               # Package initialization
└── requirements.txt          # Dependencies
```

## Algorithms Implemented

### 1. REINFORCE Agent (`algorithms.py`)

- **Core Algorithm**: Monte Carlo policy gradient
- **Features**:
  - Neural network policy with softmax output
  - Optional baseline for variance reduction
  - Experience replay buffer
  - Gradient clipping for stability
- **Key Methods**:
  - `get_action()`: Sample actions from current policy
  - `store_transition()`: Store episode transitions
  - `update_policy()`: Compute and apply policy gradients

### 2. Actor-Critic Agent (`algorithms.py`)

- **Architecture**: Separate actor (policy) and critic (value) networks
- **Features**:
  - TD(0) learning for critic updates
  - Advantage function estimation
  - Bootstrap-based policy gradients
  - Lower variance compared to REINFORCE
- **Key Methods**:
  - `get_action_and_value()`: Get action and value estimate
  - `update()`: Update both networks using TD error

### 3. Continuous Actor-Critic (`algorithms.py`)

- **Policy Type**: Gaussian policies for continuous actions
- **Features**:
  - Mean and standard deviation parameterization
  - Entropy regularization
  - Compatible with continuous control environments
- **Key Methods**:
  - `get_action()`: Sample from Gaussian distribution
  - `evaluate_action()`: Compute log probabilities and entropy

## Neural Network Architectures

### Policy Networks (`policies.py`)

- **Discrete Actions**: Multi-layer perceptron with softmax output
- **Continuous Actions**: MLP outputting mean and log standard deviation
- **Shared Features**: Actor-Critic with shared feature extraction
- **Advanced Architectures**: Batch normalization and dropout for stability

### Value Networks (`policies.py`)

- **State Value Estimation**: MLP predicting V(s)
- **Architecture**: Similar to policy networks but with scalar output

## Key Features

### Modular Design

- **Separation of Concerns**: Algorithms, policies, environments, and experiments
- **Factory Functions**: Easy algorithm instantiation
- **Configurable Networks**: Customizable hidden sizes and architectures

### Experiment Framework (`experiments.py`)

- **PolicyGradientExperiment**: Comprehensive experiment runner
- **BenchmarkSuite**: Multi-environment testing
- **Hyperparameter Sweeps**: Automated parameter optimization
- **Comparison Tools**: Algorithm performance analysis

### Visualization (`visualization.py`)

- **Policy Visualization**: Softmax and Gaussian policy plots
- **Mathematical Concepts**: Score function and baseline demonstrations
- **Training Curves**: Learning progress and loss visualization
- **Network Analysis**: Parameter counts and architecture comparisons

## Installation & Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: Neural network implementation
- **Gym/Gymnasium**: Reinforcement learning environments
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization
- **Pandas**: Data analysis

## Usage Examples

### Basic REINFORCE Training

```python
from algorithms import REINFORCEAgent
from environments import EnvironmentWrapper

# Create environment and agent
env = EnvironmentWrapper("CartPole-v1")
agent = REINFORCEAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    lr=0.001,
    baseline=True
)

# Train agent
results = agent.train(env.env, num_episodes=500)
print(f"Final score: {results['scores'][-1]:.2f}")
```

### Actor-Critic Training

```python
from algorithms import ActorCriticAgent

agent = ActorCriticAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    lr_actor=0.001,
    lr_critic=0.005
)

results = agent.train(env.env, num_episodes=500)
```

### Algorithm Comparison

```python
from experiments import PolicyGradientExperiment

experiment = PolicyGradientExperiment("CartPole-v1")
results = experiment.run_comparison_experiment(
    algorithms=["reinforce", "actor_critic"],
    num_episodes=300
)

experiment.visualize_results()
```

## Educational Content

### CA4_modular.ipynb Features

- **Policy Representations**: Understanding stochastic vs deterministic policies
- **Score Function**: Mathematical derivation of policy gradients
- **REINFORCE Implementation**: Step-by-step algorithm walkthrough
- **Actor-Critic Methods**: Temporal difference learning integration
- **Continuous Control**: Gaussian policies and entropy regularization
- **Neural Network Design**: Architecture choices and hyperparameter tuning
- **Experiment Analysis**: Performance comparison and visualization

### Key Learning Objectives

1. **Policy Gradient Theory**: Understanding direct policy optimization
2. **Variance Reduction**: Baseline subtraction and advantage functions
3. **Neural Network Policies**: Parameterizing policies with deep networks
4. **Continuous Actions**: Handling continuous action spaces
5. **Algorithm Comparison**: REINFORCE vs Actor-Critic trade-offs

## Performance & Results

### Expected Performance (CartPole-v1)

- **REINFORCE**: ~150-200 average reward (with baseline)
- **Actor-Critic**: ~180-250 average reward
- **Continuous Tasks**: Varies by environment complexity

### Training Stability

- **Gradient Clipping**: Prevents exploding gradients
- **Baseline**: Reduces variance in policy gradients
- **Entropy Bonus**: Encourages exploration in continuous spaces

## Advanced Topics

### Exploration Strategies

- **Boltzmann Exploration**: Temperature-based action selection
- **Entropy Regularization**: Intrinsic exploration bonus
- **ε-Greedy**: Simple exploration for discrete actions

### Architecture Variants

- **Shared Networks**: Parameter sharing between actor and critic
- **Batch Normalization**: Improved training stability
- **Dropout**: Regularization for complex environments

### Hyperparameter Tuning

- **Learning Rates**: Separate tuning for actor/critic
- **Network Sizes**: Hidden layer dimensions
- **Discount Factor**: Temporal credit assignment

## References & Extensions

### Related Algorithms

- **TRPO/PPO**: Trust region policy optimization
- **DDPG/SAC**: Deep deterministic policy gradients
- **A3C/A2C**: Asynchronous advantage actor-critic

### Applications

- **Robotics**: Continuous control tasks
- **Game Playing**: Complex strategy games
- **Resource Management**: Continuous decision making

## Troubleshooting

### Common Issues

- **High Variance**: Use baseline or advantage functions
- **Slow Convergence**: Adjust learning rates or network architecture
- **Instability**: Implement gradient clipping and proper initialization

### Performance Tips

- **Batch Training**: Accumulate gradients over multiple episodes
- **Target Networks**: Stabilize value function learning
- **Experience Replay**: Improve sample efficiency

---

_This assignment provides a comprehensive introduction to policy gradient methods, from basic REINFORCE to advanced Actor-Critic architectures, with practical implementations using modern deep learning frameworks._
