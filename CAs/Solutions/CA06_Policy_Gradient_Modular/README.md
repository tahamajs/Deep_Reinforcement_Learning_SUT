# CA6: Policy Gradient Methods - Modular Implementation

## Overview

This directory contains a complete, modular implementation of policy gradient methods for deep reinforcement learning. The monolithic Jupyter notebook has been refactored into organized Python modules for better maintainability, reusability, and testing.

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
CA6/
├── setup.py                    # Environment setup and utilities
├── reinforce.py                # REINFORCE algorithm
├── actor_critic.py            # Actor-Critic methods
├── advanced_pg.py             # A2C, PPO, A3C
├── variance_reduction.py      # Variance reduction techniques
├── continuous_control.py      # Continuous action spaces
├── performance_analysis.py    # Evaluation frameworks
├── applications.py            # Advanced applications
├── CA6.ipynb                  # Original monolithic notebook
├── CA6_refactored.ipynb       # Modular notebook
└── README.md                  # This documentation
```

## Contributing

When adding new features:

1. Follow the established code patterns
2. Include comprehensive logging
3. Add demonstration functions
4. Update documentation
5. Test on multiple environments

## License

This implementation is part of the Deep Reinforcement Learning course materials.
