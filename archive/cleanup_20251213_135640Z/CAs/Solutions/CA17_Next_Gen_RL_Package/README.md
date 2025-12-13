# CA17: Next-Generation Deep Reinforcement Learning Package

A comprehensive, modular implementation of cutting-edge deep reinforcement learning paradigms, transforming the monolithic CA17.ipynb notebook into a clean, importable Python package.

## 🚀 Features

This package implements state-of-the-art RL techniques across multiple paradigms:

- **World Models**: Recurrent State-Space Models (RSSM) and imagination-augmented agents
- **Multi-Agent RL**: MADDPG with communication networks for cooperative/competitive scenarios
- **Causal RL**: Causal discovery, counterfactual reasoning, and causal world models
- **Quantum-Enhanced RL**: Quantum circuits, variational quantum algorithms, and quantum state encoding
- **Federated RL**: Privacy-preserving distributed learning with differential privacy
- **Advanced Safety**: Constrained optimization, risk-sensitive learning, and real-time safety monitoring

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd CA17

# Install dependencies
pip install torch numpy gymnasium networkx scikit-learn matplotlib seaborn pandas
```

## 🏗️ Package Structure

```
CA17/
├── world_models/          # World models and imagination
├── multi_agent_rl/        # Multi-agent deep RL
├── causal_rl/            # Causal reasoning
├── quantum_rl/           # Quantum-enhanced RL
├── federated_rl/         # Federated learning
├── advanced_safety/      # Safety and robustness
├── utils/                # Utilities and helpers
├── environments/         # Custom environments
├── experiments/          # Evaluation suites
├── __init__.py          # Main package interface
└── README.md            # This file
```

## 💡 Quick Start

```python
# Import the package
import ca17

# Set random seed for reproducibility
ca17.set_random_seed(42)

# Create and run a world model experiment
config = ca17.create_default_configs()['world_model']
experiment = ca17.WorldModelExperiment(config)
results = experiment.run_experiment()

# Plot results
experiment.plot_results()
```

## 📚 Module Usage

### World Models

```python
from ca17.world_models import ImaginationAugmentedAgent, RSSMCore

# Create an imagination-augmented agent
agent = ImaginationAugmentedAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=128,
    imagination_horizon=5
)

# Train on environment
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > 64:
            agent.train_step()
        state = next_state
```

### Multi-Agent RL

```python
from ca17.multi_agent_rl import MADDPGAgent, PredatorPreyEnvironment

# Create multi-agent environment
env = PredatorPreyEnvironment(n_predators=2, n_prey=1, grid_size=10)

# Create MADDPG agent
agent = MADDPGAgent(
    n_predators=2,
    n_prey=1,
    obs_dim=env.observation_space.shape[0],
    action_dim=5
)

# Training loop
for episode in range(100):
    obs = env.reset()
    done = False
    while not done:
        actions = agent.select_actions(obs)
        next_obs, rewards, done, _ = env.step(actions)
        agent.store_transition(obs, actions, rewards, next_obs, done)
        if len(agent.replay_buffer) > 64:
            agent.train_step()
        obs = next_obs
```

### Causal RL

```python
from ca17.causal_rl import CausalRLAgent, CausalBanditEnvironment

# Create causal bandit environment
env = CausalBanditEnvironment(n_arms=3, n_contexts=2)

# Create causal RL agent
agent = CausalRLAgent(n_arms=3, n_contexts=2, hidden_dim=64)

# Training with causal discovery
for episode in range(100):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        if len(agent.replay_buffer) > 32:
            agent.train_step()
        obs = next_obs
```

### Quantum RL

```python
from ca17.quantum_rl import QuantumRLAgent, QuantumControlEnvironment

# Create quantum control environment
env = QuantumControlEnvironment(n_qubits=2, max_steps=20)

# Create quantum RL agent
agent = QuantumRLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dim=64
)

# Training loop
for episode in range(50):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        if len(agent.replay_buffer) > 32:
            agent.train_step()
        obs = next_obs
```

### Federated RL

```python
from ca17.federated_rl import FederatedRLServer, FederatedLearningEnvironment

# Create federated learning environment
env = FederatedLearningEnvironment(n_clients=10, heterogeneity=0.5)

# Create federated server
server = FederatedRLServer(n_clients=10, model_dim=1)

# Federated training rounds
for round_num in range(100):
    obs = env.reset()
    # Select participating clients
    selected_clients = np.random.choice(10, 5, replace=False)
    action = np.zeros(10)
    action[selected_clients] = 1

    # Execute federated round
    next_obs, reward, done, _, info = env.step(action)
    server.aggregate_updates(selected_clients, reward)
```

### Advanced Safety

```python
from ca17.advanced_safety import ConstrainedPolicyOptimization, SafetyMonitor

# Create safety-constrained agent
agent = ConstrainedPolicyOptimization(
    state_dim=4,
    action_dim=2,
    hidden_dim=128,
    cost_limit=1.0
)

# Create safety monitor
monitor = SafetyMonitor(
    state_dim=4,
    action_dim=2,
    safety_threshold=0.8
)

# Safe training loop
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        is_safe, cost = monitor.check_safety(state, action)
        if not is_safe:
            action = monitor.intervene(state)

        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done, cost)
        if len(agent.replay_buffer) > 64:
            agent.train_step()
        state = next_state
```

## 🧪 Running Experiments

```python
from ca17.experiments import (
    WorldModelExperiment,
    MultiAgentExperiment,
    ComparativeExperiment,
    create_default_configs
)

# Get default configurations
configs = create_default_configs()

# Run individual experiments
world_model_exp = WorldModelExperiment(configs['world_model'])
world_model_exp.run_experiment()
world_model_exp.plot_results()

# Run comparative analysis
comparative_exp = ComparativeExperiment(configs['comparative'])
comparative_exp.add_experiment('World Models', WorldModelExperiment(configs['world_model']))
comparative_exp.add_experiment('Multi-Agent', MultiAgentExperiment(configs['multi_agent']))
comparative_exp.run_experiment()
comparative_exp.plot_results()
```

## 🔧 Utilities

```python
from ca17.utils import (
    ReplayBuffer,
    Config,
    plot_learning_curve,
    compute_metrics,
    set_random_seed
)

# Configure experiments
config = Config(
    n_episodes=100,
    learning_rate=1e-3,
    batch_size=64
)

# Use replay buffer
buffer = ReplayBuffer(capacity=10000, state_dim=4, action_dim=2)

# Plot learning curves
plot_learning_curve(rewards, window=10)

# Compute performance metrics
metrics = compute_metrics(rewards)
print(f"Mean reward: {metrics['mean_reward']:.2f}")
```

## 🌍 Custom Environments

```python
from ca17.environments import (
    ContinuousMountainCar,
    CausalBanditEnvironment,
    QuantumControlEnvironment
)

# Continuous control environment
env = ContinuousMountainCar(goal_velocity=0.0)

# Causal reasoning environment
causal_env = CausalBanditEnvironment(n_arms=3, n_contexts=2)

# Quantum control environment
quantum_env = QuantumControlEnvironment(n_qubits=2)
```

## 📊 Results and Visualization

All experiments automatically generate:

- Performance metrics and statistics
- Learning curves with smoothing
- Comparative analysis plots
- Saved results in JSON format
- High-quality visualizations

## 🤝 Contributing

This package follows the modular structure established in CA15/CA16. To contribute:

1. Follow the existing code style and documentation
2. Add comprehensive tests for new features
3. Update documentation and examples
4. Ensure compatibility with existing interfaces

## 📄 License

This project is part of the DRL course assignments. Please refer to course guidelines for usage permissions.

## 🙏 Acknowledgments

This implementation is based on cutting-edge research in deep reinforcement learning, including:

- World Models (Ha & Schmidhuber, 2018)
- Multi-Agent Deep Deterministic Policy Gradient (Lowe et al., 2017)
- Causal Reasoning in RL (Buesing et al., 2018)
- Quantum Reinforcement Learning (Meyer, 2008)
- Federated Learning (McMahan et al., 2017)
- Safe Reinforcement Learning (Achiam et al., 2017)

---

For detailed API documentation, see the docstrings in each module. For questions or issues, please refer to the course materials or contact the maintainers.
