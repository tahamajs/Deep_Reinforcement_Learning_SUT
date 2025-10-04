# CA6: Policy Gradient Methods - Modular Implementation

## Overview

This directory contains a complete, modular implementation of policy gradient methods for deep reinforcement learning. The monolithic Jupyter notebook has been refactored into organized Python modules for better maintainability, reusability, and testing.

## Quick Start

```bash
# Run all algorithms and generate results
./run.sh

# Or use Python script
python main.py

# Quick mode with fewer episodes
python main.py --quick

# Run only specific components
python main.py --algorithms-only
python main.py --analyses-only
python main.py --agents-only
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make run script executable
chmod +x run.sh
```

## Modular Structure

### Core Modules

1. **`setup.py`** - Environment configuration and utilities

   - Device configuration (CPU/GPU)
   - Plotting setup
   - Random seed initialization
   - Numerical stability wrapper for `torch.distributions.Categorical`

2. **`reinforce.py`** - REINFORCE algorithm implementation

   - `REINFORCEAgent`: Complete Monte Carlo policy gradient
   - Variance analysis tools
   - Performance logging and visualization

3. **`actor_critic.py`** - Actor-Critic methods

   - `ActorCriticAgent`: Separate actor and critic networks
   - `SharedActorCriticAgent`: Shared network architecture
   - GAE (Generalized Advantage Estimation) implementation

4. **`advanced_pg.py`** - Advanced policy gradient methods

   - `A2CAgent`: Advantage Actor-Critic
   - `PPOAgent`: Proximal Policy Optimization
   - `A3CAgent`: Asynchronous Advantage Actor-Critic (with multiprocessing)

5. **`variance_reduction.py`** - Variance reduction techniques

   - `VarianceReductionAgent`: Baseline and GAE-based variance reduction
   - `ControlVariatesAgent`: Control variates for gradient variance reduction
   - Comparative analysis tools

6. **`continuous_control.py`** - Continuous action space methods

   - `ContinuousREINFORCEAgent`: REINFORCE for continuous actions
   - `ContinuousActorCriticAgent`: Actor-Critic for continuous control
   - `PPOContinuousAgent`: PPO for continuous domains
   - Gaussian policy networks

7. **`performance_analysis.py`** - Evaluation and analysis tools

   - `PolicyEvaluator`: Comprehensive policy evaluation
   - `PerformanceAnalyzer`: Learning curve and statistical analysis
   - `AblationStudy`: Parameter sensitivity analysis
   - `RobustnessTester`: Environment robustness testing

8. **`applications.py`** - Advanced applications and architectures
   - `CuriosityDrivenAgent`: Intrinsic curiosity for exploration
   - `MetaLearningAgent`: Few-shot adaptation capabilities
   - `HierarchicalAgent`: Hierarchical RL with goal selection
   - `SafeRLAgent`: Constrained policy optimization for safety

## Usage

### Running Individual Components

```python
# Import and use any agent
from reinforce import REINFORCEAgent
from actor_critic import ActorCriticAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n)

# Train for one episode
reward, loss = agent.train_episode(env)
```

### Running Complete Demonstrations

Each module includes demonstration functions:

```python
# Run full demonstrations
from reinforce import demonstrate_reinforce
from actor_critic import demonstrate_actor_critic
from advanced_pg import demonstrate_advanced_pg
from variance_reduction import demonstrate_variance_reduction
from continuous_control import demonstrate_continuous_control
from performance_analysis import demonstrate_performance_analysis
from applications import demonstrate_advanced_applications

# Run all demonstrations
reinforce_agent = demonstrate_reinforce()
ac_results = demonstrate_actor_critic()
adv_results = demonstrate_advanced_pg()
var_results, variances = demonstrate_variance_reduction()
cont_results = demonstrate_continuous_control()
analysis_report = demonstrate_performance_analysis()
app_results = demonstrate_advanced_applications()
```

### Using the Refactored Notebook

The `CA6_refactored.ipynb` notebook imports from all modules and provides a clean, organized presentation of the content without inline code duplication.

## Key Features

### Numerical Stability

- Custom wrapper for `torch.distributions.Categorical` to handle NaN/inf values
- Gradient clipping and normalization
- Stable advantage estimation

### Comprehensive Logging

- Episode rewards and losses tracking
- Gradient norm monitoring
- Entropy and exploration metrics
- Performance visualization

### Advanced Techniques

- Generalized Advantage Estimation (GAE)
- Proximal Policy Optimization (PPO)
- Control variates for variance reduction
- Curiosity-driven exploration
- Safe reinforcement learning with constraints

### Evaluation Framework

- Statistical significance testing
- Sample efficiency analysis
- Robustness testing under perturbations
- Ablation studies for parameter sensitivity

## Dependencies

- PyTorch >= 2.0
- Gymnasium >= 1.0
- NumPy >= 1.20
- Matplotlib >= 3.5
- Seaborn >= 0.11
- Pandas >= 1.3

## Environment Compatibility

All implementations are tested on:

- `CartPole-v1` (discrete actions)
- `Pendulum-v1` (continuous actions)
- Custom environments with similar interfaces

## Performance Benchmarks

Typical performance on CartPole-v1:

- REINFORCE: ~150-200 average reward
- Actor-Critic: ~180-220 average reward
- PPO: ~195+ average reward (solves environment)

## Extension Points

The modular design allows easy extension:

1. **New Algorithms**: Add new agent classes following the established patterns
2. **New Environments**: Update environment-specific parameters
3. **New Features**: Extend existing agents with new capabilities
4. **Analysis Tools**: Add new evaluation metrics and visualization

## Testing

Run individual module tests:

```bash
# Test all imports
python -c "from setup import device; from reinforce import REINFORCEAgent; from actor_critic import ActorCriticAgent; from advanced_pg import A2CAgent; from variance_reduction import VarianceReductionAgent; from continuous_control import ContinuousREINFORCEAgent; from performance_analysis import PolicyEvaluator; from applications import CuriosityDrivenAgent; print('All imports successful!')"

# Run specific demonstrations
python reinforce.py
python actor_critic.py
# etc.
```

## File Organization

```
CA06_Policy_Gradient_Modular/
├── main.py                     # Main execution script
├── run.sh                      # Bash execution script
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── CA6.ipynb                  # Original Jupyter notebook
├── training_examples.py        # Complete training examples
├──
├── agents/                     # Policy gradient agents
│   ├── reinforce.py           # REINFORCE algorithm
│   ├── actor_critic.py        # Actor-Critic methods
│   ├── advanced_pg.py         # A2C, PPO, A3C algorithms
│   └── variance_reduction.py  # Variance reduction techniques
├──
├── models/                     # Neural network models
│   ├── __init__.py
│   └── networks.py            # Policy and value networks
├──
├── environments/               # Custom environments
│   └── continuous_control.py  # Continuous control environments
├──
├── experiments/                # Advanced experiments
│   └── applications.py        # Real-world applications
├──
├── evaluation/                 # Evaluation and metrics
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics
│   └── visualization.py       # Visualization utilities
├──
├── utils/                      # Utility functions
│   ├── setup.py               # Environment setup
│   ├── performance_analysis.py # Performance analysis
│   └── run_ca6_smoke.py       # Smoke tests
├──
├── visualizations/             # Generated plots and results
├── results/                    # Training results and logs
├── logs/                       # Execution logs
└── CA6_files/                 # Original notebook outputs
```

## Features

### Implemented Algorithms

- **REINFORCE**: Monte Carlo Policy Gradient
- **REINFORCE with Baseline**: Variance reduction using value function
- **Actor-Critic**: Temporal difference learning
- **A2C (Advantage Actor-Critic)**: Synchronous advantage estimation
- **PPO (Proximal Policy Optimization)**: Policy optimization with clipping
- **A3C (Asynchronous Actor-Critic)**: Asynchronous parallel training
- **Continuous Control**: Gaussian policy for continuous actions

### Advanced Features

- **Variance Reduction**: Control variates, baseline methods
- **Curriculum Learning**: Progressive difficulty training
- **Hyperparameter Analysis**: Sensitivity analysis and tuning
- **Performance Evaluation**: Comprehensive metrics and visualization
- **Real-world Applications**: Robotics, finance, gaming examples

### Environments Supported

- **Discrete**: CartPole, LunarLander, Atari games
- **Continuous**: Pendulum, MountainCar, MuJoCo (optional)
- **Custom**: Modular environment framework

## Usage Examples

### Basic Training

```python
from training_examples import train_reinforce_agent

# Train REINFORCE on CartPole
results = train_reinforce_agent(
    env_name="CartPole-v1",
    episodes=1000,
    lr=1e-3
)
```

### Algorithm Comparison

```python
from training_examples import compare_policy_gradient_variants

# Compare all algorithms
results = compare_policy_gradient_variants(
    env_name="CartPole-v1",
    episodes=500
)
```

### Custom Agent

```python
from agents.advanced_pg import PPOAgent
import gymnasium as gym

# Create and train PPO agent
env = gym.make("CartPole-v1")
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=3e-4
)

# Training loop
for episode in range(1000):
    # ... training code ...
    pass
```

## Results and Visualizations

After running the scripts, you'll find:

- **visualizations/**: Training curves, performance comparisons, convergence analysis
- **results/**: Detailed performance metrics, comprehensive reports
- **logs/**: Execution logs and debugging information

## Performance Metrics

The framework tracks comprehensive metrics:

- **Training Performance**: Episode rewards, convergence speed
- **Sample Efficiency**: Episodes to reach target performance
- **Stability**: Variance in final performance
- **Comparison**: Relative performance across algorithms

## Contributing

To extend the framework:

1. Add new algorithms in `agents/`
2. Create custom environments in `environments/`
3. Implement new evaluation metrics in `evaluation/`
4. Add experiments in `experiments/`

## License

This project is part of the Deep Reinforcement Learning course materials.
