# CA7: Deep Q-Networks (DQN) and Value-Based Methods

## Overview

This project implements and analyzes various Deep Q-Network (DQN) algorithms, focusing on value-based reinforcement learning methods. The implementation includes basic DQN, Double DQN, and Dueling DQN, along with comprehensive analysis tools.

## Project Structure

```
CA7/
├── CA7.ipynb              # Main educational notebook
├── dqn/                   # Core DQN implementations
│   ├── __init__.py
│   ├── core.py           # Basic DQN, ReplayBuffer, DQNAgent
│   ├── double_dqn.py     # Double DQN implementation
│   ├── dueling_dqn.py    # Dueling DQN architecture
│   └── utils.py          # Visualization and analysis utilities
├── experiments/          # Experiment scripts
│   ├── __init__.py
│   ├── basic_dqn_experiment.py
│   └── ...               # Additional experiments
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Features

### Implemented Algorithms

1. **Basic DQN**

   - Experience replay buffer
   - Target networks for stability
   - ε-greedy exploration

2. **Double DQN**

   - Addresses overestimation bias
   - Decouples action selection from evaluation

3. **Dueling DQN**
   - Value-advantage decomposition
   - Separate streams for value and advantage functions

### Analysis Tools

- Performance comparison across algorithms
- Q-value distribution analysis
- Learning curve visualization
- Bias analysis and debugging tools

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support (optional):**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Running Experiments

Execute individual experiments from the `experiments/` directory:

```bash
# Basic DQN experiment
python experiments/basic_dqn_experiment.py

# Additional experiments can be added similarly
```

### Using the DQN Package

```python
from dqn import DQNAgent, DoubleDQNAgent, DuelingDQNAgent
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Create agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# Train
for episode in range(100):
    reward, steps = agent.train_episode(env)
    print(f"Episode {episode}: Reward = {reward}")

# Evaluate
results = agent.evaluate(env, num_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

### Educational Notebook

The main `CA7.ipynb` notebook provides:

- Theoretical foundations
- Step-by-step implementation walkthrough
- Interactive visualizations
- Comparative analysis

## Key Concepts Covered

### Theoretical Foundations

- Q-learning and its limitations
- Deep Q-Network architecture
- Experience replay and target networks
- Overestimation bias and solutions

### Implementation Details

- Neural network design for Q-learning
- Experience replay buffer management
- Target network updates
- Exploration strategies

### Advanced Techniques

- Double DQN for bias reduction
- Dueling architecture for better value estimation
- Performance analysis and debugging

## Results and Performance

### CartPole-v1 Environment

- **Basic DQN**: Typically achieves ~150-200 average reward
- **Double DQN**: Improved stability, ~180-220 average reward
- **Dueling DQN**: Better sample efficiency, ~200-250 average reward

### Key Insights

- Experience replay significantly improves stability
- Target networks prevent divergence
- Double DQN reduces overestimation bias
- Dueling architecture improves value estimation

## Configuration

### Hyperparameters

Default configurations work well for CartPole:

```python
agent = DQNAgent(
    lr=1e-3,                    # Learning rate
    gamma=0.99,                 # Discount factor
    epsilon_start=1.0,          # Initial exploration
    epsilon_end=0.01,           # Final exploration
    epsilon_decay=0.995,        # Exploration decay rate
    buffer_size=20000,          # Replay buffer size
    batch_size=64,              # Training batch size
    target_update_freq=100      # Target network update frequency
)
```

### Environment Setup

The code is designed to work with Gymnasium environments:

- **CartPole-v1**: Classic control, discrete actions
- **MountainCar-v0**: Continuous states, harder exploration
- **Acrobot-v1**: Underactuated system

## Contributing

To extend this project:

1. **Add new DQN variants** in the `dqn/` package
2. **Create new experiments** in the `experiments/` directory
3. **Add analysis tools** to `dqn/utils.py`
4. **Update the notebook** with new findings

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) - Double DQN
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) - Dueling DQN
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) - Rainbow DQN

## License

This project is part of the Deep Reinforcement Learning course materials.
