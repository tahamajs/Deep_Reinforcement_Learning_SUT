# Reinforcement Learning GridWorld Implementation

This project contains a complete, modular implementation of reinforcement learning algorithms for the GridWorld environment. The code has been refactored from a monolithic Jupyter notebook into well-organized Python modules for better maintainability and reusability.

## Project Structure

```
CA2/
├── __init__.py              # Package initialization and exports
├── environments.py          # GridWorld environment implementation
├── policies.py              # Policy classes (Random, Greedy, Custom, etc.)
├── algorithms.py            # Core RL algorithms (Policy Iteration, Value Iteration, Q-Learning)
├── visualization.py         # Plotting and visualization functions
├── experiments.py           # Experiment functions for systematic testing
├── CA2_modular.ipynb        # Clean notebook that imports from modules
└── README.md               # This file
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install numpy matplotlib seaborn pandas
```

## Quick Start

### Using the Modular Implementation

```python
from environments import GridWorld
from policies import RandomPolicy
from algorithms import policy_evaluation
from visualization import plot_value_function

# Create environment
env = GridWorld()

# Create and evaluate a policy
policy = RandomPolicy(env)
values = policy_evaluation(env, policy, gamma=0.9)

# Visualize results
plot_value_function(env, values, "Random Policy Values")
```

### Running Experiments

```python
from experiments import run_all_experiments

# Run all experiments with default settings
run_all_experiments()
```

### Using Individual Components

```python
# Import specific components
from environments import create_custom_environment
from policies import create_policy
from algorithms import policy_iteration

# Create custom environment
env = create_custom_environment(size=5, obstacles=[(1,1), (2,2)])

# Create policy using factory function
policy = create_policy('greedy')

# Run policy iteration
optimal_policy, optimal_values, history = policy_iteration(env, gamma=0.9)
```

## Modules Overview

### environments.py

- `GridWorld`: Main environment class with customizable rewards, obstacles, and transitions
- `create_custom_environment()`: Factory function for creating modified environments

### policies.py

- `Policy`: Abstract base class for all policies
- `RandomPolicy`: Uniform random action selection
- `GreedyPolicy`: Greedy action selection based on Q-values
- `CustomPolicy`: Custom policy with action preferences
- `GreedyActionPolicy`: Greedy policy using value functions
- `create_policy()`: Factory function for creating policies

### algorithms.py

- `policy_evaluation()`: Evaluate a policy to compute value function
- `compute_q_from_v()`: Compute Q-values from value function
- `compute_v_from_q()`: Compute value function from Q-values
- `policy_iteration()`: Policy iteration algorithm
- `value_iteration()`: Value iteration algorithm
- `q_learning()`: Q-learning algorithm

### visualization.py

- `plot_value_function()`: Visualize value functions as heatmaps
- `plot_policy()`: Visualize policies with action arrows
- `plot_q_values()`: Visualize Q-values for all state-action pairs
- `plot_learning_curve()`: Plot learning progress over episodes
- `compare_policies()`: Compare multiple policies side-by-side

### experiments.py

- `experiment_discount_factors()`: Test different discount factors
- `experiment_policy_comparison()`: Compare different policies
- `experiment_policy_iteration()`: Run policy iteration with visualization
- `experiment_value_iteration()`: Run value iteration with convergence plots
- `experiment_q_learning()`: Train agent with Q-learning
- `experiment_environment_modifications()`: Test different environment configurations
- `run_all_experiments()`: Run all experiments in sequence

## Key Features

1. **Modular Design**: Clean separation of concerns with dedicated modules for each component
2. **Extensible Architecture**: Easy to add new environments, policies, and algorithms
3. **Comprehensive Visualization**: Rich plotting capabilities for analysis and debugging
4. **Educational Focus**: Well-documented code with clear explanations and examples
5. **Experiment Framework**: Systematic testing of different parameters and configurations

## Usage Examples

### Basic Policy Evaluation

```python
from environments import GridWorld
from policies import RandomPolicy
from algorithms import policy_evaluation

env = GridWorld()
policy = RandomPolicy(env)
values = policy_evaluation(env, policy, gamma=0.9)

print(f"Value of start state: {values[(0,0)]:.3f}")
```

### Policy Iteration

```python
from algorithms import policy_iteration
from visualization import plot_policy, plot_value_function

optimal_policy, optimal_values, history = policy_iteration(env, gamma=0.9)

plot_value_function(env, optimal_values, "Optimal Value Function")
plot_policy(env, optimal_policy, "Optimal Policy")
```

### Q-Learning Training

```python
from algorithms import q_learning
from visualization import plot_learning_curve

Q, episode_rewards = q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)
plot_learning_curve(episode_rewards, "Q-Learning Progress")
```

## Educational Value

This implementation serves as an excellent foundation for learning reinforcement learning concepts:

- **MDP Fundamentals**: Clear implementation of states, actions, transitions, and rewards
- **Value Functions**: Both state-value (V) and action-value (Q) functions
- **Dynamic Programming**: Policy evaluation, policy iteration, and value iteration
- **Temporal Difference Learning**: Q-learning implementation
- **Policy Types**: Different policy strategies and their implications

## Contributing

To extend this implementation:

1. **New Environments**: Add new environment classes in `environments.py`
2. **New Policies**: Implement new policy classes inheriting from `Policy`
3. **New Algorithms**: Add new algorithm functions in `algorithms.py`
4. **New Visualizations**: Add plotting functions in `visualization.py`
5. **New Experiments**: Add experiment functions in `experiments.py`

## License

This project is provided for educational purposes. Feel free to use and modify the code for learning and research.
