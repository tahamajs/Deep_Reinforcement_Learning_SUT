# CA10: Model-Based Reinforcement Learning and Planning Methods

## Overview

This assignment explores **Model-Based Reinforcement Learning (MBRL)**, focusing on methods that learn explicit models of the environment and use them for planning and decision making. The implementation covers classical planning algorithms, integrated planning-learning approaches, and modern neural model-based methods.

## Key Concepts

### Model-Based vs Model-Free RL

- **Model-Free**: Learn policies/values directly from experience
- **Model-Based**: Learn environment model first, then plan using the model
- **Hybrid Approaches**: Combine both paradigms for best of both worlds

### Core Components

- **Model Learning**: Learning transition dynamics and reward functions
- **Planning**: Using learned models to compute optimal policies
- **Integration**: Combining planning with learning for improved sample efficiency

## Project Structure

```
CA10/
├── CA10.ipynb                    # Main educational notebook
├── classical_planning.py         # Classical planning algorithms
├── dyna_q.py                     # Dyna-Q and integrated planning-learning
├── mcts.py                       # Monte Carlo Tree Search
├── mpc.py                        # Model Predictive Control
├── models.py                     # Environment model implementations
├── environments.py               # Custom test environments
├── comparison.py                 # Comparative analysis tools
├── visualizations/               # Generated plots and analysis
└── requirements.txt              # Dependencies
```

## Core Algorithms

### 1. Classical Planning (`classical_planning.py`)

- **Value Iteration**: Dynamic programming with learned models
- **Policy Iteration**: Alternating policy evaluation and improvement
- **Uncertainty-Aware Planning**: Pessimistic and optimistic planning
- **Model-Based Policy Search**: Random shooting and cross-entropy methods

### 2. Dyna-Q Algorithm (`dyna_q.py`)

- **Dyna-Q**: Integrated planning and learning
- **Dyna-Q+**: Prioritized sweeping for efficient planning
- **Experience Replay**: Storing and reusing experience
- **Planning Steps**: Configurable planning-to-learning ratio

### 3. Monte Carlo Tree Search (`mcts.py`)

- **MCTS Core**: Selection, expansion, simulation, backpropagation
- **UCB1 Selection**: Upper confidence bound for exploration
- **Neural MCTS**: Combining MCTS with neural networks
- **Applications**: Game playing and general RL domains

### 4. Model Predictive Control (`mpc.py`)

- **MPC Framework**: Receding horizon control
- **Gradient-Based Optimization**: Efficient trajectory optimization
- **Sampling-Based MPC**: Cross-entropy method for MPC
- **Constrained MPC**: Handling state and action constraints

### 5. Environment Models (`models.py`)

- **Tabular Models**: Explicit transition and reward tables
- **Neural Models**: Function approximation for complex dynamics
- **Ensemble Models**: Multiple models for uncertainty quantification
- **Model Training**: Maximum likelihood estimation and neural training

## Key Features

### Modular Architecture

- **Clean Separation**: Models, algorithms, and environments in separate modules
- **Extensible Design**: Easy to add new algorithms or models
- **Reproducible Results**: Fixed random seeds and comprehensive logging
- **Educational Focus**: Detailed comments and theoretical explanations

### Comprehensive Analysis Tools

- **Performance Metrics**: Learning curves, sample efficiency, final performance
- **Uncertainty Quantification**: Model confidence and prediction intervals
- **Planning Analysis**: Planning time, horizon effects, model accuracy
- **Comparative Studies**: Model-based vs model-free, different planning methods

### Custom Environments

- **SimpleGridWorld**: Basic grid navigation with obstacles
- **BlockingMaze**: Dynamic environment with changing obstacles
- **Continuous Control Tasks**: Pendulum and cart-pole variants

## Installation & Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: Neural network implementations and optimization
- **NumPy**: Numerical computations and data structures
- **Gymnasium**: Reinforcement learning environments
- **Matplotlib/Seaborn**: Visualization and plotting
- **NetworkX**: Graph operations for planning algorithms

## Usage Examples

### Basic Model Learning

```python
from models import TabularModel, NeuralModel
from environments import SimpleGridWorld

# Create environment
env = SimpleGridWorld(size=5)

# Learn tabular model
tabular_model = TabularModel(env.num_states, env.num_actions)
# Collect experience and update model...

# Train neural model
neural_model = NeuralModel(env.num_states, env.num_actions, hidden_dim=64)
# Training loop...
```

### Classical Planning

```python
from classical_planning import ModelBasedPlanner

# Create planner with learned model
planner = ModelBasedPlanner(tabular_model, env.num_states, env.num_actions)

# Value iteration
values, policy = planner.value_iteration(max_iterations=100)

# Policy iteration
values, policy = planner.policy_iteration(max_iterations=50)
```

### Dyna-Q Learning

```python
from dyna_q import DynaQAgent

# Create Dyna-Q agent
agent = DynaQAgent(
    state_dim=env.num_states,
    action_dim=env.num_actions,
    planning_steps=10  # Number of planning updates per real step
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

### Monte Carlo Tree Search

```python
from mcts import MCTSAgent

# Create MCTS agent
agent = MCTSAgent(
    state_dim=env.num_states,
    action_dim=env.num_actions,
    num_simulations=1000,
    c_puct=1.0  # Exploration parameter
)

# Planning from current state
action = agent.search(state)
```

### Model Predictive Control

```python
from mpc import MPCAgent

# Create MPC agent
agent = MPCAgent(
    model=neural_model,
    horizon=20,  # Planning horizon
    num_samples=100  # Number of trajectory samples
)

# Get optimal action sequence
actions, values = agent.plan(state)
best_action = actions[0]  # Execute first action
```

## Educational Content

### CA10.ipynb Structure

1. **Theoretical Foundations**: Model-based vs model-free comparison
2. **Environment Models**: Tabular and neural model learning
3. **Classical Planning**: Value/policy iteration with learned models
4. **Dyna-Q**: Integrating planning and learning
5. **MCTS**: Tree search for planning
6. **MPC**: Optimal control with learned models
7. **Advanced Methods**: Modern neural approaches
8. **Comparative Analysis**: Performance evaluation and insights

### Key Learning Objectives

1. **Model Learning**: Understanding different model representations
2. **Planning Algorithms**: Classical DP and modern sampling methods
3. **Sample Efficiency**: How model-based methods improve data usage
4. **Uncertainty Handling**: Dealing with model imperfections
5. **Integration Strategies**: Combining planning and learning effectively
6. **Advanced Applications**: MCTS, MPC, and neural planning

## Performance & Results

### Sample Efficiency Comparison

- **Model-Free (Q-Learning)**: ~1000-2000 episodes for convergence
- **Dyna-Q (planning_steps=5)**: ~300-600 episodes for convergence
- **Dyna-Q (planning_steps=50)**: ~100-200 episodes for convergence

### Planning Method Performance

- **Value Iteration**: Fast convergence, requires accurate models
- **MCTS**: Robust to model errors, computationally expensive
- **MPC**: Good for continuous control, handles constraints
- **Dyna-Q**: Best sample efficiency, balances computation and learning

### Model Accuracy Impact

- **High Accuracy Models**: Planning methods outperform model-free
- **Medium Accuracy Models**: Hybrid approaches (Dyna-Q) perform best
- **Low Accuracy Models**: Model-free methods more robust

## Advanced Topics

### Uncertainty Quantification

- **Ensemble Models**: Multiple models for confidence estimation
- **Bayesian Neural Networks**: Probabilistic model predictions
- **Pessimistic Planning**: Conservative planning under uncertainty

### Modern Neural Methods

- **World Models**: End-to-end learning of dynamics and rewards
- **Dreamer**: Planning in learned latent spaces
- **MuZero**: Learning models, values, and policies jointly

### Scalability Challenges

- **Model Learning**: Computational cost of learning accurate models
- **Planning Complexity**: Exponential growth with planning horizon
- **Real-World Deployment**: Handling partial observability and non-stationarity

## Applications

### Robotics

- **Motion Planning**: Trajectory optimization with dynamics models
- **Manipulation**: Learning object interaction models
- **Locomotion**: Model-based control for walking and running

### Game Playing

- **Board Games**: MCTS for Go, Chess, and strategy games
- **Video Games**: Model-based agents for complex environments
- **Real-Time Strategy**: Hierarchical planning for resource management

### Autonomous Systems

- **Self-Driving Cars**: MPC for trajectory planning
- **Drone Control**: Model-based flight planning and control
- **Industrial Automation**: Predictive control for manufacturing

### Healthcare

- **Treatment Planning**: Modeling patient response to interventions
- **Drug Discovery**: Planning molecular design sequences
- **Personalized Medicine**: Individual patient models

## Troubleshooting

### Common Issues

- **Model Inaccuracy**: Poor planning performance with bad models
- **Computational Cost**: Planning too slow for real-time applications
- **Model Bias**: Systematic errors in learned dynamics
- **Exploration**: Insufficient exploration in model-based planning

### Best Practices

- **Model Validation**: Always validate models on held-out data
- **Planning Horizons**: Balance horizon length with computational cost
- **Uncertainty Awareness**: Use ensemble or Bayesian models
- **Hybrid Approaches**: Combine model-based and model-free methods

### Performance Tips

- **Prioritized Sweeping**: Focus planning on promising states
- **Approximate Planning**: Use sampling instead of exact DP
- **Model Ensembles**: Multiple models reduce prediction variance
- **Adaptive Planning**: Adjust planning effort based on model confidence

---

_This assignment provides a comprehensive exploration of model-based reinforcement learning, from classical planning algorithms to modern neural approaches, with practical implementations and thorough analysis tools._
