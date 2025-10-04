# CA18 - Advanced Reinforcement Learning Paradigms

A modular Python package implementing cutting-edge reinforcement learning algorithms including quantum-enhanced RL, world models, multi-agent systems, causal RL, federated learning, and advanced safety mechanisms.

## Overview

This package transforms the monolithic CA18.ipynb notebook into a clean, maintainable, and reusable codebase. It implements advanced RL paradigms that go beyond traditional approaches:

- **Quantum RL**: Quantum-enhanced reinforcement learning with variational circuits
- **World Models**: Model-based RL with Recurrent State-Space Models (RSSM)
- **Multi-Agent RL**: Cooperative/competitive multi-agent systems with communication
- **Causal RL**: Causal reasoning and intervention in reinforcement learning
- **Federated RL**: Privacy-preserving distributed learning
- **Advanced Safety**: Robust policies with quantum-inspired uncertainty quantification

## Installation

### Quick Start (Recommended)
```bash
# Make the script executable and run it
chmod +x run.sh
./run.sh
```

This will automatically:
- Create a virtual environment
- Install all dependencies  
- Run comprehensive demos
- Generate visualizations and reports

### Manual Installation
```bash
# Create virtual environment
python3 -m venv ca18_env
source ca18_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_modules.py
```

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Quick Start

```python
from CA18 import quantum_rl, causal_rl, experiments

# Create a quantum RL agent
agent = quantum_rl.QuantumQLearning(
    state_dim=4,
    action_dim=2,
    n_qubits=4
)

# Create experiment
experiment = experiments.QuantumRLExperiment(
    agent_class=quantum_rl.QuantumQLearning,
    environment_class=quantum_rl.QuantumEnvironment
)

# Run experiment
results = experiment.run_experiment()
```

## Package Structure

```
CA18/
├── quantum_rl/          # Quantum-enhanced RL algorithms
├── world_models/        # World model-based RL
├── multi_agent_rl/      # Multi-agent systems
├── causal_rl/           # Causal RL with discovery
├── federated_rl/        # Federated learning
├── advanced_safety/     # Safety constraints & robust policies
├── utils/               # Advanced utilities & data structures
├── environments/        # Test environments
├── experiments/         # Experiment frameworks
└── __init__.py         # Main package interface
```

## Key Components

### Quantum RL Module

```python
from CA18.quantum_rl import QuantumQLearning, QuantumActorCritic

# Q-learning with quantum states
agent = QuantumQLearning(state_dim=8, action_dim=4, n_qubits=4)

# Actor-critic with quantum circuits
agent = QuantumActorCritic(state_dim=8, action_dim=4, n_qubits=4)
```

### Causal RL Module

```python
from CA18.causal_rl import CausalDiscovery, CausalWorldModel

# Causal structure discovery
discovery = CausalDiscovery(n_variables=5)
graph = discovery.discover_causal_structure(data)

# Causal world model
world_model = CausalWorldModel(state_dim=10, action_dim=4, causal_graph=graph)
```

### Multi-Agent RL Module

```python
from CA18.multi_agent_rl import MADDPGAgent, MultiAgentEnvironment

# Multi-agent environment
env = MultiAgentEnvironment(n_agents=3, state_dim=8, action_dim=4)

# MADDPG agent
agent = MADDPGAgent(
    n_agents=3,
    state_dim=8,
    action_dim=4,
    communication_dim=16
)
```

### Advanced Safety Module

```python
from CA18.advanced_safety import QuantumConstrainedPolicyOptimization

# Safe policy optimization with quantum regularization
agent = QuantumConstrainedPolicyOptimization(
    state_dim=8,
    action_dim=4,
    cost_limit=0.1,
    quantum_reg_weight=0.1
)
```

## Experiment Frameworks

### Running Comparative Experiments

```python
from CA18.experiments import ComparativeExperimentRunner

# Define algorithms to compare
algorithms = {
    'quantum_rl': {
        'experiment_class': experiments.QuantumRLExperiment,
        'agent_class': quantum_rl.QuantumQLearning,
        'agent_kwargs': {'n_qubits': 4}
    },
    'causal_rl': {
        'experiment_class': experiments.CausalRLExperiment,
        'agent_class': causal_rl.CausalPolicyGradient,
        'agent_kwargs': {'causal_graph': graph}
    }
}

# Run comparison
runner = ComparativeExperimentRunner()
runner.run_comparison(algorithms, environment_class, n_runs=3)
```

### Custom Experiments

```python
from CA18.experiments import BaseExperiment

class CustomExperiment(BaseExperiment):
    def run_experiment(self):
        # Implement custom experiment logic
        pass

experiment = CustomExperiment("my_experiment")
results = experiment.run_experiment()
experiment.plot_results()
```

## Advanced Features

### Quantum-Inspired Utilities

```python
from CA18.utils import QuantumPrioritizedReplayBuffer, QuantumRNG

# Quantum prioritized replay buffer
buffer = QuantumPrioritizedReplayBuffer(capacity=10000, quantum_dim=8)

# Quantum random number generator
rng = QuantumRNG()
random_action = rng.quantum_choice(actions)
```

### Specialized Environments

```python
from CA18.environments import QuantumEnvironment, CausalBanditEnvironment

# Quantum control environment
env = QuantumEnvironment(n_qubits=4, max_steps=100)

# Causal bandit with hidden structure
env = CausalBanditEnvironment(n_arms=5, n_context_vars=3)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- NetworkX
- scikit-learn
- matplotlib
- seaborn
- pandas

## Citation

If you use this package in your research, please cite:

```
@software{CA18_Advanced_RL,
  title = {CA18: Advanced Reinforcement Learning Paradigms},
  author = {CA18 Development Team},
  year = {2024},
  url = {https://github.com/your-repo/CA18}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
