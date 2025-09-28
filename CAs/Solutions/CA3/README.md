# Temporal Difference Learning Implementation

This project contains a comprehensive implementation of temporal difference (TD) learning algorithms, including TD(0), Q-Learning, SARSA, and various exploration strategies. The code demonstrates the transition from model-based to model-free reinforcement learning using both monolithic and modular implementations.

## Project Structure

```
CA3/
├── __init__.py              # Package initialization
├── CA3.ipynb               # Main notebook with theory, implementation, and experiments
├── CA3_modular.ipynb       # Clean modular notebook using separate modules
├── algorithms.py           # Core TD learning algorithms (TD0, Q-Learning, SARSA)
├── environments.py         # GridWorld environment implementation
├── exploration.py          # Exploration strategies and experiments
├── experiments.py          # Systematic experiment functions
├── policies.py             # Policy classes and implementations
├── visualization.py        # Plotting and analysis functions
└── requirements.txt        # Python dependencies
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install numpy matplotlib seaborn pandas
```

Or using the requirements file (if populated):

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Q-Learning Agent

```python
from environments import GridWorld
from algorithms import QLearningAgent

# Create environment
env = GridWorld()

# Create and train Q-Learning agent
agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
agent.train(num_episodes=1000)

# Evaluate learned policy
evaluation = agent.evaluate_policy(num_episodes=100)
print(f"Average reward: {evaluation['avg_reward']:.2f}")
print(f"Success rate: {evaluation['success_rate']*100:.1f}%")
```

### Comparing TD Algorithms

```python
from algorithms import TD0Agent, QLearningAgent, SARSAAgent
from policies import RandomPolicy
from experiments import compare_algorithms

# Create agents
env = GridWorld()
random_policy = RandomPolicy(env)

agents = {
    'TD(0)': TD0Agent(env, random_policy),
    'Q-Learning': QLearningAgent(env),
    'SARSA': SARSAAgent(env)
}

# Compare performance
results = compare_algorithms(agents, env)
```

### Testing Exploration Strategies

```python
from exploration import ExplorationExperiment

# Test different exploration strategies
strategies = {
    'epsilon_0.1': {'epsilon': 0.1, 'decay': 1.0},
    'epsilon_decay': {'epsilon': 0.9, 'decay': 0.99},
    'boltzmann': {'temperature': 2.0}
}

experiment = ExplorationExperiment(env)
results = experiment.run_exploration_experiment(strategies)
```

## Modules Overview

### environments.py

- **`GridWorld`**: Custom grid-based environment with obstacles, rewards, and configurable parameters
- Support for episodic interaction with standard RL environment interface
- Visualization capabilities for value functions and policies

### algorithms.py

- **`TD0Agent`**: TD(0) algorithm for policy evaluation
- **`QLearningAgent`**: Off-policy Q-Learning for optimal control
- **`SARSAAgent`**: On-policy SARSA algorithm
- All agents include training loops, evaluation methods, and learning curve tracking

### policies.py

- **`RandomPolicy`**: Uniform random action selection
- **`GreedyPolicy`**: Greedy action selection based on Q-values
- **`CustomPolicy`**: Configurable policy with action preferences
- Factory functions for easy policy creation

### exploration.py

- **`ExplorationStrategies`**: Collection of exploration methods (ε-greedy, Boltzmann, decay schedules)
- **`ExplorationExperiment`**: Systematic comparison of exploration strategies
- **`BoltzmannQLearning`**: Q-Learning with Boltzmann exploration

### experiments.py

- **`compare_algorithms()`**: Comprehensive comparison of TD algorithms
- **`run_exploration_experiment()`**: Exploration strategy analysis
- Statistical analysis and visualization of results

### visualization.py

- **`plot_value_function()`**: Visualize state values as heatmaps
- **`plot_policy()`**: Display learned policies with action arrows
- **`plot_learning_curve()`**: Learning progress over episodes
- **`compare_policies()`**: Side-by-side policy comparison

## Key Features

### 1. Complete TD Learning Suite

- **TD(0)**: Policy evaluation without model knowledge
- **Q-Learning**: Off-policy control for optimal policies
- **SARSA**: On-policy control for behavior policies
- Bootstrap learning from experience without complete episodes

### 2. Exploration Strategies

- **ε-greedy**: Classic exploration with configurable ε
- **Decaying ε**: Adaptive exploration that reduces over time
- **Boltzmann Exploration**: Probabilistic action selection based on Q-values
- Systematic comparison and analysis tools

### 3. Comprehensive Analysis

- Learning curve visualization and analysis
- Statistical performance comparison
- Value function and policy visualization
- Hyperparameter sensitivity analysis

### 4. Educational Implementation

- Clear separation between theory and implementation
- Detailed comments explaining TD concepts
- Modular design for easy experimentation
- Both monolithic and modular code versions

## Usage Examples

### Basic TD(0) Learning

```python
from environments import GridWorld
from policies import RandomPolicy
from algorithms import TD0Agent

env = GridWorld()
policy = RandomPolicy(env)
agent = TD0Agent(env, policy, alpha=0.1, gamma=0.9)

V = agent.train(num_episodes=500)
print(f"Learned value of start state: {V[(0,0)]:.3f}")
```

### Q-Learning Training

```python
from algorithms import QLearningAgent
from visualization import plot_value_function, plot_policy

agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
agent.train(num_episodes=1000)

V_optimal = agent.get_value_function()
optimal_policy = agent.get_policy()

plot_value_function(env, V_optimal, "Optimal Value Function")
plot_policy(env, optimal_policy, "Optimal Policy")
```

### Algorithm Comparison

```python
import matplotlib.pyplot as plt
from experiments import compare_algorithms

# Train all algorithms
results = compare_algorithms({
    'TD(0)': TD0Agent(env, RandomPolicy(env)),
    'Q-Learning': QLearningAgent(env),
    'SARSA': SARSAAgent(env)
}, env)

# Plot comparison
plt.figure(figsize=(12, 4))
for name, result in results.items():
    plt.plot(result['rewards'], label=name, alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Algorithm Comparison')
plt.legend()
plt.show()
```

### Exploration Strategy Analysis

```python
from exploration import ExplorationExperiment

strategies = {
    'Fixed ε=0.1': {'epsilon': 0.1, 'decay': 1.0},
    'Decaying ε': {'epsilon': 0.9, 'decay': 0.99},
    'Boltzmann τ=2.0': {'temperature': 2.0}
}

experiment = ExplorationExperiment(env)
performance = experiment.run_exploration_experiment(strategies)

# Print results
for strategy, stats in performance.items():
    print(f"{strategy}: Avg Reward = {stats['mean_reward']:.2f}")
```

## Theoretical Foundations

### Temporal Difference Learning

- **Bootstrapping**: Update estimates using current estimates
- **Online Learning**: Learn from incomplete episodes
- **TD Error**: Prediction error driving learning updates
- **Convergence**: Conditions for TD methods to converge

### Control Algorithms

- **Q-Learning**: Learns optimal policy through off-policy learning
- **SARSA**: Learns policy being followed (on-policy)
- **Exploration-Exploitation Tradeoff**: Balancing learning and performance
- **Convergence Properties**: Different convergence guarantees

### Exploration Strategies

- **ε-greedy**: Simple but effective exploration
- **Boltzmann**: Principled probabilistic exploration
- **Decay Schedules**: Adaptive exploration over time
- **Performance Tradeoffs**: Different strategies for different environments

## Experimental Results

The implementation includes comprehensive experiments comparing:

1. **Algorithm Performance**: TD(0) vs Q-Learning vs SARSA
2. **Exploration Strategies**: Fixed ε, decaying ε, Boltzmann
3. **Hyperparameter Sensitivity**: Learning rates, discount factors, exploration rates
4. **Learning Dynamics**: Convergence behavior and stability

### Key Findings

- **Q-Learning**: Most aggressive, learns optimal policy fastest
- **SARSA**: More conservative, safer in dangerous environments
- **TD(0)**: Foundation for control, good for policy evaluation
- **Exploration**: Critical for discovering good policies
- **Hyperparameters**: Significant impact on learning speed and final performance

## Educational Value

This implementation serves as a comprehensive resource for learning temporal difference learning:

- **Algorithm Understanding**: Clear implementations of core TD methods
- **Practical Skills**: Experience with exploration strategies and hyperparameter tuning
- **Comparative Analysis**: Systematic evaluation of different approaches
- **Theoretical Connections**: Links between TD theory and implementation
- **Research Skills**: Designing and analyzing RL experiments

## Dependencies

- **NumPy**: Numerical computations and arrays
- **Matplotlib/Seaborn**: Visualization and plotting
- **Pandas**: Data manipulation and analysis

## Contributing

To extend this implementation:

1. **New Environments**: Add new environment classes in `environments.py`
2. **New Algorithms**: Implement new TD variants in `algorithms.py`
3. **New Exploration**: Add exploration strategies in `exploration.py`
4. **New Experiments**: Create new experimental setups in `experiments.py`
5. **New Visualizations**: Add plotting functions in `visualization.py`

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
- Watkins, C. J. C. H. (1989). Learning from delayed rewards. PhD thesis, University of Cambridge.
- Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems. University of Cambridge.

## License

This project is provided for educational purposes. Feel free to use and modify the code for learning and research.</content>
<parameter name="filePath">/Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA3/README.md
