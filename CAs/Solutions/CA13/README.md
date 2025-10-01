# CA13: Advanced Deep Reinforcement Learning - Model-Free vs Model-Based Methods and Real-World Applications

## Overview

This assignment explores advanced deep reinforcement learning concepts, focusing on the fundamental differences between model-free and model-based approaches, world models, sample efficiency techniques, and practical considerations for real-world deployment. The notebook provides comprehensive implementations and comparisons of cutting-edge RL methods.

**Status**: ✅ Complete and ready to run

## Learning Objectives

1. **Model-Free vs Model-Based RL**: Understand theoretical foundations, trade-offs, and practical implementations
2. **World Models**: Learn environment dynamics using VAE-based architectures for imagination-based planning
3. **Sample Efficiency**: Master techniques like prioritized replay, data augmentation, and auxiliary tasks
4. **Transfer Learning**: Implement knowledge reuse and adaptation across related tasks
5. **Hierarchical RL**: Explore temporal abstraction using Options-Critic and Feudal Networks
6. **Comprehensive Evaluation**: Compare methods across multiple dimensions and provide practical guidelines

## Key Concepts Covered

### 1. Model-Free vs Model-Based RL

- **Model-Free Methods**: DQN, policy gradients - learn directly from experience
- **Model-Based Methods**: Learn environment dynamics for planning and imagination
- **Hybrid Approaches**: Dyna-Q style learning combining both paradigms
- **Trade-off Analysis**: Sample efficiency vs computational complexity

### 2. World Models and Imagination

- **Variational Autoencoders (VAE)**: Stochastic latent representations
- **Dynamics Modeling**: Predicting next states and rewards
- **Imagination-Based Planning**: Planning in learned latent space
- **Dreamer Algorithm**: Combining world models with policy learning

### 3. Sample Efficiency Techniques

- **Prioritized Experience Replay**: Focus on important transitions
- **Data Augmentation**: Robustness through input transformations
- **Auxiliary Tasks**: Multi-task learning for better representations
- **Curriculum Learning**: Progressive difficulty for stable learning

### 4. Transfer Learning and Meta-Learning

- **Transfer Learning**: Leveraging knowledge from source to target tasks
- **Fine-tuning**: Adapting pre-trained models to new domains
- **Meta-Learning**: Learning to learn across multiple tasks

### 5. Hierarchical Reinforcement Learning

- **Options Framework**: Temporal abstractions as skills
- **Options-Critic**: Joint learning of options and policies
- **Feudal Networks**: Manager-worker hierarchies with intrinsic motivation
- **Hindsight Experience Replay**: Goal-conditioned learning

## Project Structure

```
CA13/
├── CA13.ipynb                 # Main notebook with implementations
├── README.md                  # This documentation
├── agents/                    # Modular agent implementations
│   ├── __init__.py
│   ├── model_free.py         # DQN, Policy Gradient agents
│   ├── model_based.py        # Dynamics model, MPC agents
│   ├── hybrid.py             # Dyna-Q, hybrid approaches
│   ├── world_model.py        # VAE-based world models
│   ├── imagination.py        # Imagination-based planning
│   ├── hierarchical.py       # Options-Critic, Feudal networks
│   └── transfer.py           # Transfer learning agents
├── environments/              # Custom environments
│   ├── __init__.py
│   ├── gridworld.py          # Simple grid navigation
│   └── utils.py              # Environment utilities
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── buffers.py            # Experience replay buffers
│   ├── evaluation.py         # Evaluation frameworks
│   └── visualization.py      # Plotting and analysis
└── results/                   # Experiment results and logs
    ├── experiments/          # Saved experiment data
    └── plots/               # Generated visualizations
```

## Installation and Setup

### Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn pandas
pip install plotly gym gymnasium
pip install scikit-learn jupyter
```

### Quick Start

```python
# Import key components
from agents.model_free import DQNAgent
from agents.model_based import ModelBasedAgent
from agents.world_model import VariationalWorldModel
from environments.gridworld import SimpleGridWorld

# Create environment and agents
env = SimpleGridWorld(size=5)
model_free_agent = DQNAgent(state_dim=2, action_dim=4)
model_based_agent = ModelBasedAgent(state_dim=2, action_dim=4)

# Compare performance
results = compare_agents_performance(env, [model_free_agent, model_based_agent])
```

## Key Implementations

### Core Agent Classes

#### Model-Free Agents

```python
class DQNAgent:
    """Deep Q-Network with experience replay"""
    def __init__(self, state_dim, action_dim, lr=1e-3)
    def act(self, state, epsilon=0.1)  # Epsilon-greedy action selection
    def update(self, batch)            # DQN learning update
```

#### Model-Based Agents

```python
class ModelBasedAgent:
    """Learns dynamics model for planning"""
    def __init__(self, state_dim, action_dim, planning_horizon=5)
    def update_model(self, batch)      # Update dynamics/reward models
    def plan_action(self, state)       # MPC-style planning
```

#### World Model Agents

```python
class VariationalWorldModel:
    """VAE-based world model"""
    def encode(self, obs)              # Observation to latent
    def dynamics_forward(self, z, a)   # Predict next latent
    def imagine_trajectory(self, obs, actions)  # Rollout in imagination
```

#### Hierarchical Agents

```python
class OptionsCriticAgent:
    """Options-Critic architecture"""
    def select_option(self, state)     # Choose high-level option
    def select_action(self, state, option)  # Execute within option
    def update(self, trajectory)       # Joint option-policy learning
```

### Advanced Techniques

#### Sample Efficiency

- **PrioritizedReplayBuffer**: TD-error based prioritization
- **DataAugmentationDQN**: Input transformations for robustness
- **Auxiliary Tasks**: Reward and dynamics prediction

#### Transfer Learning

- **TransferLearningAgent**: Multi-task policy heads
- **CurriculumLearningFramework**: Progressive difficulty
- **Fine-tuning**: Knowledge transfer between tasks

## Usage Examples

### Basic Comparison

```python
# Compare model-free vs model-based
env = SimpleGridWorld(size=5)
agents = {
    'DQN': DQNAgent(2, 4),
    'Model-Based': ModelBasedAgent(2, 4),
    'Hybrid': HybridDynaAgent(2, 4)
}

results = compare_agents_performance(env, agents)
visualize_comparison(results)
```

### World Model Learning

```python
# Train world model for imagination
world_model = VariationalWorldModel(obs_dim=2, action_dim=4)
agent = ImaginationBasedAgent(obs_dim=2, action_dim=4)

# Train and evaluate
trained_agent, rewards, losses = demonstrate_world_model_learning()
visualize_world_model_performance(trained_agent, rewards, losses)
```

### Hierarchical Learning

```python
# Options-Critic for temporal abstraction
agent = OptionsCriticAgent(state_dim=2, action_dim=4, num_options=4)

# Train with hierarchical learning
results, agents = demonstrate_hierarchical_rl()
```

### Transfer Learning

```python
# Transfer between related tasks
agent = TransferLearningAgent(state_dim=2, action_dim=4)
agent.add_task('task1')
agent.add_task('task2')

# Fine-tune from task1 to task2
agent.fine_tune_for_task('task1', 'task2')
```

## Results and Analysis

### Performance Comparison

The notebook provides comprehensive evaluation across:

- **Sample Efficiency**: Episodes to convergence
- **Final Performance**: Asymptotic reward levels
- **Robustness**: Performance variance across runs
- **Transfer Capability**: Adaptation to new tasks

### Key Findings

1. **Model-Based methods** excel in sample efficiency but struggle with high-dimensional observations
2. **World models** enable imagination-based planning and improve exploration
3. **Hierarchical methods** provide temporal abstractions for long-horizon tasks
4. **Sample efficiency techniques** (prioritized replay, augmentation) significantly improve learning
5. **Transfer learning** enables rapid adaptation to related domains

## Applications and Extensions

### Real-World Applications

- **Robotics**: Model-based control for manipulation and navigation
- **Game AI**: World models for strategic planning in complex games
- **Autonomous Systems**: Safe exploration with imagination-based methods
- **Healthcare**: Sample-efficient learning from limited patient data

### Extensions

- **Multi-Agent RL**: Coordination with hierarchical policies
- **Meta-RL**: Learning across distributions of tasks
- **Offline RL**: Learning from fixed datasets
- **Safe RL**: Constrained optimization for safety-critical domains

## Educational Value

This assignment provides:

- **Theoretical Understanding**: Deep dive into advanced RL concepts
- **Practical Implementation**: Working code for all major techniques
- **Comparative Analysis**: Empirical evaluation of different approaches
- **Research Insights**: Understanding of current RL research directions

## References

1. **World Models**: Ha & Schmidhuber (2018) - Recurrent World Models Facilitate Policy Evolution
2. **Options Framework**: Sutton et al. (1999) - Between MDPs and Semi-MDPs
3. **Dreamer**: Hafner et al. (2019) - Learning to Simulate and Dream
4. **Options-Critic**: Bacon et al. (2017) - The Option-Critic Architecture
5. **Feudal Networks**: Vezhnevets et al. (2017) - Feudal Networks for Hierarchical Reinforcement Learning

## Next Steps

After completing CA13, you should be able to:

- Choose appropriate RL methods for different problem domains
- Implement advanced techniques for sample-efficient learning
- Design hierarchical policies for complex tasks
- Apply transfer learning for knowledge reuse
- Deploy RL systems with practical considerations

This comprehensive assignment bridges the gap between theoretical RL concepts and real-world applications, preparing you for advanced research and industry applications in deep reinforcement learning.</content>
<parameter name="filePath">/Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA13/README.md
