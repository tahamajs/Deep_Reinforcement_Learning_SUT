# Deep Reinforcement Learning Implementation

This project contains a comprehensive implementation of deep reinforcement learning algorithms, including Deep Q-Networks (DQN), REINFORCE, and Actor-Critic methods. The code demonstrates both theoretical foundations and practical implementations using PyTorch and OpenAI Gym environments.

## Project Structure

```
CA1/
├── __init__.py              # Package initialization
├── CA1.ipynb               # Main notebook with theory, implementation, and experiments
├── ca1_agents.py           # RL agent implementations (DQN, REINFORCE, Actor-Critic)
├── ca1_models.py           # Neural network architectures (DQN, DuelingDQN, Policy Networks)
├── ca1_utils.py            # Utility functions for training and visualization
└── requirements.txt        # Python dependencies
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install gymnasium matplotlib seaborn numpy pandas
```

Or using the requirements file (if populated):

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a DQN Agent

```python
import gymnasium as gym
from ca1_agents import DQNAgent, train_dqn_agent

# Create environment
env = gym.make('CartPole-v1')

# Create and train DQN agent
agent = DQNAgent(state_size=4, action_size=2, use_dueling=True, use_double_dqn=True)
scores = train_dqn_agent(agent, env, n_episodes=1000)

print(f"Training completed. Final average score: {np.mean(scores[-100:]):.2f}")
```

### Training a Policy Gradient Agent

```python
from ca1_agents import REINFORCEAgent, train_reinforce_agent

# Create REINFORCE agent
reinforce_agent = REINFORCEAgent(state_size=4, action_size=2)
scores = train_reinforce_agent(reinforce_agent, env, n_episodes=1000)

print(f"REINFORCE training completed. Final average score: {np.mean(scores[-100:]):.2f}")
```

### Training an Actor-Critic Agent

```python
from ca1_agents import ActorCriticAgent, train_actor_critic_agent

# Create Actor-Critic agent
ac_agent = ActorCriticAgent(state_size=4, action_size=2)
scores = train_actor_critic_agent(ac_agent, env, n_episodes=1000)

print(f"Actor-Critic training completed. Final average score: {np.mean(scores[-100:]):.2f}")
```

## Modules Overview

### ca1_agents.py

Contains implementations of all reinforcement learning agents:

- **`DQNAgent`**: Deep Q-Network agent with experience replay, target networks, and optional Double DQN and Dueling DQN extensions
- **`REINFORCEAgent`**: Monte Carlo policy gradient agent using REINFORCE algorithm
- **`ActorCriticAgent`**: Actor-Critic agent with separate policy and value networks
- **`ReplayBuffer`**: Experience replay buffer for DQN training
- **Training Functions**: `train_dqn_agent()`, `train_reinforce_agent()`, `train_actor_critic_agent()`

### ca1_models.py

Neural network architectures:

- **`DQN`**: Standard Deep Q-Network with fully connected layers
- **`DuelingDQN`**: Dueling architecture separating value and advantage streams
- **`PolicyNetwork`**: Policy network for discrete action spaces
- **`ValueNetwork`**: Value function approximator for Actor-Critic
- **`NoisyLinear`**: Noisy linear layer for exploration in DQN
- **`NoisyDQN`**: DQN with noisy layers for intrinsic exploration

### ca1_utils.py

Utility functions and helpers:

- **`set_seed()`**: Set random seeds for reproducibility
- **`moving_average()`**: Compute moving averages for plotting
- **`gym_reset()` / `gym_step()`**: Handle different Gym API versions
- Device configuration and plotting style setup

## Key Features

### 1. Multiple DQN Variants

- **Standard DQN**: Basic implementation with experience replay and target networks
- **Double DQN**: Reduces overestimation bias by decoupling action selection and evaluation
- **Dueling DQN**: Separates value and advantage estimation for better performance
- **Noisy DQN**: Uses parameter noise for exploration instead of ε-greedy

### 2. Policy Gradient Methods

- **REINFORCE**: Monte Carlo policy gradient with variance reduction
- **Actor-Critic**: Combines policy gradient with value function approximation
- **Baseline subtraction**: Reduces variance in policy gradient estimates

### 3. Comprehensive Training Framework

- Automated training loops with progress tracking
- Environment solving detection (CartPole-v1 threshold: 195.0)
- Hyperparameter tuning support
- Sample efficiency experiments

### 4. Educational Components

- Detailed theoretical explanations in the notebook
- Implementation of key RL concepts (MDPs, value functions, Bellman equations)
- Comparative analysis of different algorithms
- Hyperparameter sensitivity analysis

## Usage Examples

### Comparing Algorithms

```python
from ca1_agents import DQNAgent, REINFORCEAgent, ActorCriticAgent
from ca1_utils import moving_average
import matplotlib.pyplot as plt

# Train all agents
agents = {
    'DQN': DQNAgent(4, 2, use_dueling=True, use_double_dqn=True),
    'REINFORCE': REINFORCEAgent(4, 2),
    'Actor-Critic': ActorCriticAgent(4, 2)
}

results = {}
for name, agent in agents.items():
    if name == 'DQN':
        scores = train_dqn_agent(agent, env, n_episodes=500)
    elif name == 'REINFORCE':
        scores = train_reinforce_agent(agent, env, n_episodes=500)
    else:
        scores = train_actor_critic_agent(agent, env, n_episodes=500)
    results[name] = scores

# Plot comparison
plt.figure(figsize=(12, 8))
for name, scores in results.items():
    plt.plot(moving_average(scores, 50), label=name)
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Algorithm Comparison on CartPole-v1')
plt.legend()
plt.show()
```

### Hyperparameter Tuning

```python
# Test different learning rates
learning_rates = [1e-4, 1e-3, 1e-2]
results = {}

for lr in learning_rates:
    agent = DQNAgent(state_size=4, action_size=2, lr=lr)
    scores = train_dqn_agent(agent, env, n_episodes=200)
    results[f'lr={lr}'] = np.mean(scores[-50:])

print("Learning Rate Comparison:")
for config, score in results.items():
    print(f"{config}: {score:.2f}")
```

### Custom Environment

```python
# Use different Gym environments
environments = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']

for env_name in environments:
    try:
        env = gym.make(env_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(state_size, action_size)
        scores = train_dqn_agent(agent, env, n_episodes=300)

        print(f"{env_name}: Best score = {max(scores):.2f}, Avg last 50 = {np.mean(scores[-50:]):.2f}")
        env.close()
    except Exception as e:
        print(f"Error with {env_name}: {e}")
```

## Theoretical Foundations

### Markov Decision Processes (MDPs)

The notebook covers the mathematical framework of MDPs including:

- State and action spaces
- Transition probabilities and reward functions
- Discount factors and return calculations
- Bellman equations for value functions

### Value-Based Methods

- State-value and action-value functions
- Q-learning and Deep Q-Networks
- Experience replay and target networks
- Exploration strategies (ε-greedy, noisy networks)

### Policy-Based Methods

- Policy gradient theorem derivation
- REINFORCE algorithm implementation
- Actor-Critic architecture
- Variance reduction techniques

## Experimental Results

The implementation includes comprehensive experiments comparing:

1. **Sample Efficiency**: How quickly each algorithm learns
2. **Stability**: Training stability and convergence properties
3. **Performance**: Final performance on CartPole-v1 benchmark
4. **Hyperparameter Sensitivity**: Impact of learning rates, discount factors, etc.

### Key Findings

- **DQN**: Most sample efficient, stable training, best final performance
- **Actor-Critic**: Good balance of sample efficiency and stability
- **REINFORCE**: Higher variance, requires more episodes to converge

## Advanced Features

### Prioritized Experience Replay

```python
from ca1_agents import PrioritizedReplayBuffer

# Use prioritized replay for improved sample efficiency
prioritized_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
```

### Custom Network Architectures

```python
# Create custom DQN with different hidden sizes
custom_dqn = DQN(state_size=4, action_size=2, hidden_size=128)
```

### Multi-Environment Support

The code is designed to work with any Gym environment with discrete actions and continuous state spaces.

## Educational Value

This implementation serves as a comprehensive resource for learning deep reinforcement learning:

- **Algorithm Understanding**: Clear implementations of core DRL algorithms
- **Practical Skills**: PyTorch implementation patterns for RL
- **Experimental Design**: Proper evaluation and comparison methodologies
- **Theoretical Connections**: Links between theory and practice
- **Best Practices**: Modern DRL implementation techniques

## Dependencies

- **PyTorch**: Deep learning framework
- **Gymnasium**: Reinforcement learning environments (modern Gym fork)
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization and plotting
- **Pandas**: Data manipulation (for analysis)

## Contributing

To extend this implementation:

1. **New Algorithms**: Add new agent classes in `ca1_agents.py`
2. **New Architectures**: Add neural network models in `ca1_models.py`
3. **New Environments**: Test with additional Gym environments
4. **Analysis Tools**: Add new visualization or analysis functions
5. **Experiments**: Create new experimental setups

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv preprint.
- Hasselt, H. V., et al. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.

## License

This project is provided for educational purposes. Feel free to use and modify the code for learning and research.</content>
<parameter name="filePath">/Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA1/README.md
