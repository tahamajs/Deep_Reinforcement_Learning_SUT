# CA5 Module Import Guide

Quick reference for importing and using CA5 DQN implementations.

---

## Core DQN Agents

### Standard DQN
```python
from agents.dqn_base import DQN, DQNAgent, ReplayBuffer

# Create agent
agent = DQNAgent(
    state_size=4,
    action_size=2,
    lr=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    buffer_size=10000,
    batch_size=32,
    target_update_freq=1000
)

# Train agent
scores, losses = agent.train(env, num_episodes=500, print_every=100)

# Use trained agent
action = agent.get_action(state, epsilon=0.01)
```

---

### Double DQN
```python
from agents.double_dqn import DoubleDQNAgent, OverestimationAnalysis

# Create Double DQN agent (reduces overestimation bias)
agent = DoubleDQNAgent(
    state_size=4,
    action_size=2,
    lr=0.001,
    gamma=0.99
)

# Analyze overestimation
analysis = OverestimationAnalysis()
results = analysis.visualize_bias_analysis()
```

---

### Dueling DQN
```python
from agents.dueling_dqn import DuelingDQNAgent, DuelingAnalysis

# Create Dueling DQN agent (separates V and A)
agent = DuelingDQNAgent(
    state_size=4,
    action_size=2,
    lr=0.001
)

# Analyze value decomposition
analyzer = DuelingAnalysis()
analyzer.visualize_decomposition(agent, env, num_episodes=10)
```

---

### Prioritized Experience Replay
```python
from agents.prioritized_replay import PrioritizedDQNAgent, PriorityAnalysis

# Create agent with prioritized replay
agent = PrioritizedDQNAgent(
    state_size=4,
    action_size=2,
    alpha=0.6,      # Priority exponent
    beta=0.4,       # IS correction
    epsilon=1e-6
)

# Analyze priorities
analyzer = PriorityAnalysis()
analyzer.plot_priority_distribution()
```

---

### Rainbow DQN (All Improvements)
```python
from agents.rainbow_dqn import RainbowDQNAgent

# Create Rainbow DQN agent (state-of-the-art)
agent = RainbowDQNAgent(
    state_size=4,
    action_size=2,
    n_step=3,           # Multi-step learning
    n_atoms=51,         # Distributional RL
    v_min=-10,
    v_max=10,
    lr=0.001
)

# Train (combines all improvements)
scores, losses = agent.train(env, num_episodes=500)
```

---

## Analysis & Visualization Tools

### Architecture Analysis
```python
from utils.network_architectures import (
    DQNArchitectureComparison,
    analyze_dqn_architectures
)

# Compare different architectures
comparison = DQNArchitectureComparison(
    state_size=4,
    action_size=2
)

# Analyze all architectures
results = analyze_dqn_architectures(
    env_name='CartPole-v1',
    architectures=['standard', 'conv', 'dueling']
)
```

---

### Training Analysis
```python
from utils.training_analysis import DQNAnalysis

# Comprehensive training analysis
analyzer = DQNAnalysis(
    agent=agent,
    training_scores=scores,
    training_losses=losses
)

# Plot training progress
analyzer.plot_training_progress()

# Analyze learning dynamics
analyzer.analyze_learning_dynamics()

# Create summary report
analyzer.create_summary_report(save_path='report.pdf')
```

---

## Advanced Extensions

### Huber Loss for Robustness
```python
from utils.advanced_dqn_extensions import (
    DoubleDQNHuberAgent,
    analyze_loss_functions
)

# Agent with Huber loss (robust to outliers)
agent = DoubleDQNHuberAgent(
    state_size=4,
    action_size=2,
    huber_delta=1.0  # Transition point
)

# Visualize loss comparison
analyze_loss_functions()
```

---

### Novelty-Based Prioritization
```python
from utils.advanced_dqn_extensions import (
    NoveltyEstimator,
    NoveltyPrioritizedReplayBuffer,
    NoveltyPriorityDebugger
)

# Estimate state novelty
estimator = NoveltyEstimator(
    state_dim=4,
    method='hybrid',  # count, neural, knn, or hybrid
    k_neighbors=5
)

# Buffer with novelty + TD error
buffer = NoveltyPrioritizedReplayBuffer(
    capacity=10000,
    state_dim=4,
    alpha_td=0.6,        # Weight for TD error
    alpha_novelty=0.4    # Weight for novelty
)

# Debug priorities
debugger = NoveltyPriorityDebugger(buffer)
debugger.plot_priority_components()
debugger.analyze_sampling_bias()
```

---

### Multi-Objective DQN
```python
from utils.advanced_dqn_extensions import (
    MultiObjectiveDQNAgent,
    MultiObjectiveEnvironment
)

# Agent for multiple objectives
agent = MultiObjectiveDQNAgent(
    state_size=2,
    action_size=4,
    num_objectives=3,
    scalarization='linear',  # linear, chebyshev, lexicographic, pareto
    objective_weights=[0.5, 0.3, 0.2]
)

# Multi-objective environment
env = MultiObjectiveEnvironment(grid_size=10)
state = env.reset()
next_state, rewards, done = env.step(action)  # rewards is a list

# Visualize Pareto front
sample_states = [np.random.randn(2) for _ in range(100)]
agent.plot_pareto_front(sample_states)
```

---

## Common Usage Patterns

### Training Loop
```python
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Choose and create agent
from agents.rainbow_dqn import RainbowDQNAgent
agent = RainbowDQNAgent(state_size, action_size)

# Train
scores, losses = agent.train(
    env=env,
    num_episodes=1000,
    print_every=100
)

# Evaluate
from utils.training_analysis import DQNAnalysis
analyzer = DQNAnalysis(agent, scores, losses)
analyzer.plot_training_progress()
```

---

### Comparison Study
```python
from agents.dqn_base import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.prioritized_replay import PrioritizedDQNAgent
from agents.rainbow_dqn import RainbowDQNAgent

# Create all agents
agents = {
    'Standard DQN': DQNAgent(state_size, action_size),
    'Double DQN': DoubleDQNAgent(state_size, action_size),
    'Dueling DQN': DuelingDQNAgent(state_size, action_size),
    'Prioritized': PrioritizedDQNAgent(state_size, action_size),
    'Rainbow': RainbowDQNAgent(state_size, action_size)
}

# Train and compare
results = {}
for name, agent in agents.items():
    print(f"Training {name}...")
    scores, _ = agent.train(env, num_episodes=500)
    results[name] = scores

# Visualize comparison
import matplotlib.pyplot as plt
for name, scores in results.items():
    plt.plot(scores, label=name, alpha=0.7)
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Variants Comparison')
plt.show()
```

---

## Module Structure

```
agents/
├── dqn_base.py              # Standard DQN
├── double_dqn.py            # Double DQN + analysis
├── dueling_dqn.py           # Dueling architecture
├── prioritized_replay.py   # Prioritized experience replay
└── rainbow_dqn.py           # All improvements combined

utils/
├── ca5_helpers.py                  # Helper functions
├── analysis_tools.py               # Basic analysis
├── network_architectures.py       # Architecture comparison
├── training_analysis.py            # Training visualization
└── advanced_dqn_extensions.py     # Advanced features
```

---

## Hyperparameter Tuning Tips

### Standard DQN
- **Learning rate**: Start with 0.001, try [0.0001, 0.001, 0.01]
- **Gamma**: 0.99 for most tasks
- **Epsilon decay**: 0.995 (slower) to 0.99 (faster)
- **Target update**: 1000-2000 steps
- **Batch size**: 32 or 64

### Double DQN
- Same as Standard DQN
- Slightly higher learning rate may work better

### Dueling DQN
- Same as Standard DQN
- May need slightly lower learning rate (more parameters)

### Prioritized Replay
- **Alpha**: 0.6 (higher = more prioritization)
- **Beta**: Start at 0.4, anneal to 1.0
- **Epsilon**: 1e-6 (small constant for priorities)

### Rainbow DQN
- **N-step**: 3-5 steps
- **N-atoms**: 51 (distributional RL)
- **V-min/max**: Based on environment rewards
- More sensitive to hyperparameters

---

## Troubleshooting

### Agent not learning
1. Check learning rate (try different values)
2. Increase buffer size
3. Check target network update frequency
4. Verify reward scaling

### Training unstable
1. Use Huber loss instead of MSE
2. Reduce learning rate
3. Increase target update frequency
4. Clip gradients (already done in most agents)

### Slow convergence
1. Try prioritized replay
2. Increase batch size
3. Adjust epsilon decay
4. Use Rainbow DQN

### Memory issues
1. Reduce buffer size
2. Reduce batch size
3. Use smaller network architecture

---

## Quick Tips

✅ **Start simple**: Use Standard DQN first, then add improvements  
✅ **Monitor training**: Use visualization tools regularly  
✅ **Save checkpoints**: Save model weights during training  
✅ **Experiment**: Try different hyperparameters systematically  
✅ **Visualize**: Use analysis tools to understand agent behavior  

---

Happy Reinforcement Learning! 🚀
