# CA7 Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd CAs/Solutions/CA7

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Notebook

```bash
# Start Jupyter
jupyter notebook CA7.ipynb
```

The notebook is organized into sections that build upon each other:
- **Sections I-III**: Theoretical foundations and setup
- **Sections IV-VII**: Core DQN implementations and variants
- **Section VIII**: Comprehensive comparisons
- **Section IX**: Conclusions and references
- **Section X**: Advanced analysis experiments
- **Section XI**: Implementation guidelines

### 3. Using the Python Modules

#### Basic DQN Training

```python
from agents.core import DQNAgent
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Create agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=1e-3,
    gamma=0.99,
    epsilon_decay=0.995
)

# Train
for episode in range(100):
    reward, steps = agent.train_episode(env)
    if (episode + 1) % 25 == 0:
        print(f"Episode {episode+1}: Reward = {reward}")

# Evaluate
results = agent.evaluate(env, num_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

#### Double DQN

```python
from agents.double_dqn import DoubleDQNAgent

agent = DoubleDQNAgent(
    state_dim=4,
    action_dim=2,
    lr=1e-3,
    epsilon_decay=0.995
)

# Training same as basic DQN
```

#### Dueling DQN

```python
from agents.dueling_dqn import DuelingDQNAgent

agent = DuelingDQNAgent(
    state_dim=4,
    action_dim=2,
    dueling_type='mean',  # 'mean', 'max', or 'naive'
    lr=1e-3
)
```

### 4. Running Experiments

#### Basic DQN Experiment

```bash
python experiments/basic_dqn_experiment.py
```

This will:
- Train a basic DQN agent on CartPole-v1
- Display training progress
- Show performance plots
- Analyze Q-value distributions

#### Comprehensive Analysis

```bash
python experiments/comprehensive_dqn_analysis.py
```

This will:
- Compare all DQN variants
- Analyze experience replay strategies
- Test different hyperparameters
- Generate comprehensive plots and reports

### 5. Visualization and Analysis

#### Q-Learning Concepts Visualization

```python
from agents.utils import QNetworkVisualization

visualizer = QNetworkVisualization()
visualizer.visualize_q_learning_concepts()
visualizer.demonstrate_overestimation_bias()
```

#### Performance Analysis

```python
from agents.utils import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze Q-value distributions
analyzer.analyze_q_value_distributions(agent, env, num_samples=1000)

# Compare learning curves
results = {
    'DQN': {'rewards': rewards_dqn, 'losses': losses_dqn},
    'Double DQN': {'rewards': rewards_ddqn, 'losses': losses_ddqn}
}
PerformanceAnalyzer.plot_learning_curves(results)
```

## Key Features

### 1. Modular Architecture

All code is organized into logical modules:
- `agents/core.py`: Core DQN components
- `agents/double_dqn.py`: Double DQN extension
- `agents/dueling_dqn.py`: Dueling DQN architecture
- `agents/utils.py`: Visualization and analysis tools

### 2. Comprehensive Documentation

Every module, class, and function includes:
- Detailed docstrings
- Type hints
- Usage examples
- Mathematical formulations

### 3. Flexible Configuration

Easy hyperparameter tuning:

```python
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    lr=1e-3,                  # Learning rate
    gamma=0.99,               # Discount factor
    epsilon_start=1.0,        # Initial exploration
    epsilon_end=0.01,         # Final exploration
    epsilon_decay=0.995,      # Exploration decay
    buffer_size=10000,        # Replay buffer size
    batch_size=64,            # Training batch size
    target_update_freq=100    # Target network update frequency
)
```

### 4. Multiple Environments

Test on various Gym environments:

```python
environments = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']

for env_name in environments:
    env = gym.make(env_name)
    # Train agent...
```

## Common Issues and Solutions

### Issue: Import errors

**Solution**: Ensure you're running from the CA7 directory or have added it to your Python path:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
```

### Issue: Slow training

**Solution**: Reduce number of episodes or use GPU:

```python
# Check GPU availability
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Issue: Poor performance

**Solutions**:
1. Increase training episodes
2. Tune hyperparameters (especially learning rate)
3. Increase replay buffer size
4. Try different DQN variants

## Best Practices

### 1. Start Simple

Begin with basic DQN before trying advanced variants:

```python
# Start here
agent = DQNAgent(state_dim=4, action_dim=2)

# Then try
agent = DoubleDQNAgent(state_dim=4, action_dim=2)

# Finally
agent = DuelingDQNAgent(state_dim=4, action_dim=2)
```

### 2. Monitor Training

Track multiple metrics:

```python
print(f"Episode {episode}")
print(f"  Reward: {reward}")
print(f"  Epsilon: {agent.epsilon}")
print(f"  Buffer size: {len(agent.replay_buffer)}")
print(f"  Loss: {agent.losses[-1] if agent.losses else 'N/A'}")
```

### 3. Save Your Work

Save trained agents:

```python
# Save
agent.save('my_agent.pth')

# Load
agent.load('my_agent.pth')
```

### 4. Experiment Systematically

Use consistent evaluation:

```python
def evaluate_agent(agent, env, num_episodes=10):
    results = agent.evaluate(env, num_episodes=num_episodes)
    print(f"Mean: {results['mean_reward']:.2f}")
    print(f"Std: {results['std_reward']:.2f}")
    return results
```

## Advanced Usage

### Custom Environments

```python
import gymnasium as gym
from gymnasium import spaces

class MyCustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,))
        self.action_space = spaces.Discrete(2)
    
    def reset(self):
        # Return initial state
        return np.random.random(4), {}
    
    def step(self, action):
        # Return next_state, reward, terminated, truncated, info
        return np.random.random(4), 1.0, False, False, {}

# Use with DQN
env = MyCustomEnv()
agent = DQNAgent(state_dim=4, action_dim=2)
```

### Custom Networks

```python
from agents.core import DQN
import torch.nn as nn

class CustomQNetwork(DQN):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim, hidden_dims=[512, 512, 256])
        # Add custom layers or modifications
```

## Performance Benchmarks

Expected performance on CartPole-v1 (500 episodes):

| Algorithm | Mean Reward | Std | Training Time |
|-----------|-------------|-----|---------------|
| Basic DQN | 180-220 | 20-30 | ~30-60s |
| Double DQN | 200-240 | 15-25 | ~35-65s |
| Dueling DQN | 210-250 | 10-20 | ~40-70s |

*Times are approximate and depend on hardware*

## Additional Resources

- **Notebook**: `CA7.ipynb` - Complete educational notebook with all experiments
- **Training Examples**: `training_examples.py` - Standalone training examples
- **Experiments**: `experiments/` - Comprehensive analysis scripts
- **README**: `README.md` - Project overview and setup
- **Changes**: `CHANGES.md` - Detailed changelog

## Support

For issues or questions:
1. Check the notebook examples
2. Review the module docstrings
3. Run the experiment scripts
4. Consult the README.md

## Citation

If you use this code for academic work, please cite:

```bibtex
@misc{ca7_dqn,
  title={Deep Q-Networks and Value-Based Methods: A Comprehensive Implementation},
  author={CA7 Implementation},
  year={2024},
  howpublished={\url{https://github.com/yourusername/DRL}}
}
```

