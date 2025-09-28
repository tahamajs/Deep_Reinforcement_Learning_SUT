# CA19 Modular: Advanced RL Systems Package

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.7+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A modular Python package implementing next-generation reinforcement learning systems from CA19, featuring hybrid quantum-classical algorithms, neuromorphic intelligence, and quantum-enhanced control systems.

## 🚀 Features

- **Hybrid Quantum-Classical RL**: Fusion of quantum circuits with classical neural networks
- **Neuromorphic RL**: Brain-inspired learning with spiking neural networks and STDP plasticity
- **Quantum RL**: Quantum-enhanced agents for complex control tasks
- **Advanced Environments**: Specialized testbeds for evaluating different RL paradigms
- **Comprehensive Experiments**: Systematic evaluation and comparison frameworks
- **Modular Design**: Clean separation of concerns for easy experimentation and extension

## 📦 Installation

### Prerequisites

```bash
# Required dependencies
pip install numpy torch matplotlib seaborn

# Optional dependencies (for full functionality)
pip install qiskit scipy
```

### Install Package

```bash
# Clone or download the CA19_modular package
cd /path/to/CA19_modular
pip install -e .
```

## 🏗️ Package Structure

```
CA19_modular/
├── hybrid_quantum_classical_rl/    # Hybrid quantum-classical agents
├── neuromorphic_rl/               # Brain-inspired neuromorphic learning
├── quantum_rl/                    # Quantum-enhanced RL systems
├── environments/                  # Advanced RL environments
├── experiments/                   # Experiment frameworks
├── utils/                         # Configuration and utilities
└── __init__.py                    # Main package interface
```

## 🎯 Quick Start

### Basic Usage

```python
from ca19_modular import (
    HybridQuantumClassicalAgent,
    NeuromorphicEnvironment,
    run_demo_experiment
)

# Run a quick demo
results = run_demo_experiment(
    episodes=20,
    agent_type='hybrid',
    env_type='neuromorphic'
)

print(f"Average reward: {results['stats']['avg_reward']:.2f}")
```

### Advanced Experiment

```python
from ca19_modular import (
    QuantumNeuromorphicComparison,
    MissionConfig,
    ExperimentRunner
)

# Configure experiment
config = MissionConfig(
    state_dim=8,
    action_dim=16,
    max_episodes=100
)

# Run comprehensive comparison
experiment = QuantumNeuromorphicComparison(config)
results = experiment.run_comparison_experiment(
    num_episodes=50,
    max_steps=200
)

# Generate report
experiment.plot_comparison_results('comparison_results.png')
print(experiment.generate_performance_report())
```

## 📚 Modules Overview

### Hybrid Quantum-Classical RL (`hybrid_quantum_classical_rl`)

Fusion of quantum circuits with classical neural networks for enhanced representation learning.

```python
from ca19_modular.hybrid_quantum_classical_rl import HybridQuantumClassicalAgent

agent = HybridQuantumClassicalAgent(
    state_dim=10,
    action_dim=8,
    quantum_dim=6,
    hidden_dim=64
)
```

**Key Classes:**

- `HybridQuantumClassicalAgent`: Main agent class
- `QuantumStateSimulator`: Quantum state encoding
- `QuantumFeatureMap`: Feature mapping circuits
- `VariationalQuantumCircuit`: Parameterized quantum circuits

### Neuromorphic RL (`neuromorphic_rl`)

Brain-inspired learning with spiking neural networks and spike-timing-dependent plasticity.

```python
from ca19_modular.neuromorphic_rl import NeuromorphicActorCritic

agent = NeuromorphicActorCritic(
    state_dim=12,
    action_dim=4,
    neuron_count=50,
    synapse_count=200
)
```

**Key Classes:**

- `NeuromorphicActorCritic`: Main neuromorphic agent
- `SpikingNeuron`: Leaky integrate-and-fire neurons
- `STDPSynapse`: Plastic synapses with STDP
- `SpikingNetwork`: Event-driven neural networks

### Quantum RL (`quantum_rl`)

Quantum-enhanced reinforcement learning for complex control tasks.

```python
from ca19_modular.quantum_rl import QuantumEnhancedAgent, SpaceStationEnvironment

agent = QuantumEnhancedAgent(
    state_dim=15,
    action_dim=20,
    quantum_dim=8
)

env = SpaceStationEnvironment(difficulty_level="EXTREME")
```

**Key Classes:**

- `QuantumEnhancedAgent`: Quantum-enhanced agent
- `QuantumRLCircuit`: Variational quantum circuits for RL
- `SpaceStationEnvironment`: Critical infrastructure control
- `MissionTrainer`: Training system with performance analysis

### Environments (`environments`)

Specialized environments for testing advanced RL paradigms.

```python
from ca19_modular.environments import (
    NeuromorphicEnvironment,
    HybridQuantumClassicalEnvironment,
    MetaLearningEnvironment
)

# Event-driven neuromorphic environment
env1 = NeuromorphicEnvironment(state_dim=6, action_dim=4)

# Complex hybrid environment
env2 = HybridQuantumClassicalEnvironment(
    state_dim=12,
    action_dim=16,
    quantum_complexity=0.8
)

# Meta-learning environment
env3 = MetaLearningEnvironment(
    base_state_dim=8,
    num_tasks=5
)
```

### Experiments (`experiments`)

Comprehensive evaluation and comparison frameworks.

```python
from ca19_modular.experiments import (
    QuantumNeuromorphicComparison,
    AblationStudy,
    ScalabilityAnalysis
)

# Full comparison experiment
comparison = QuantumNeuromorphicComparison(config)
results = comparison.run_comparison_experiment()

# Ablation studies
ablation = AblationStudy(config)
quantum_results = ablation.run_quantum_ablation(env)
neuromorphic_results = ablation.run_neuromorphic_ablation(env)

# Scalability analysis
scalability = ScalabilityAnalysis(config)
scale_results = scalability.run_scalability_test()
```

### Utilities (`utils`)

Configuration management, performance tracking, and helper functions.

```python
from ca19_modular.utils import (
    MissionConfig,
    PerformanceTracker,
    ExperimentManager,
    benchmark_quantum_vs_classical
)

# Configuration
config = MissionConfig(
    n_qubits=6,
    difficulty_level="EXTREME",
    quantum_weight_adaptive=True
)

# Performance tracking
tracker = PerformanceTracker()
# ... training loop ...
tracker.update_episode(reward, length, metrics)
stats = tracker.get_summary_stats()

# Benchmarking
results = benchmark_quantum_vs_classical(
    quantum_agent, classical_agent, environment
)
```

## 🔬 Research Applications

### Quantum Advantage Studies

- Compare quantum vs classical performance
- Analyze entanglement effects on learning
- Study quantum state preparation efficiency

### Neuromorphic Intelligence

- Event-driven learning in dynamic environments
- Energy-efficient computation models
- Biological plausibility validation

### Hybrid Systems

- Optimal quantum-classical integration
- Adaptive computation allocation
- Multi-paradigm learning strategies

## 📊 Example Experiments

### 1. Quantum vs Classical Benchmark

```python
from ca19_modular import benchmark_quantum_vs_classical

# Setup agents and environment
quantum_agent = HybridQuantumClassicalAgent(...)
classical_agent = ClassicalAgent(...)  # Your classical baseline
environment = SpaceStationEnvironment()

# Run benchmark
results = benchmark_quantum_vs_classical(
    quantum_agent, classical_agent, environment,
    num_episodes=100
)

print(f"Quantum advantage: {results['advantage_percent']:.1f}%")
```

### 2. Scalability Analysis

```python
from ca19_modular.experiments import ScalabilityAnalysis

analysis = ScalabilityAnalysis(config)
results = analysis.run_scalability_test(
    problem_sizes=[4, 8, 16, 32, 64]
)

analysis.plot_scalability_results('scalability.png')
```

### 3. Ablation Studies

```python
from ca19_modular.experiments import AblationStudy

study = AblationStudy(config)

# Test quantum components
quantum_ablation = study.run_quantum_ablation(
    environment, num_episodes=50
)

# Test neuromorphic components
neuro_ablation = study.run_neuromorphic_ablation(
    environment, num_episodes=50
)
```

## 🔧 Configuration

### MissionConfig Parameters

```python
config = MissionConfig(
    # Quantum circuit parameters
    n_qubits=6,                    # Number of qubits
    n_layers=3,                    # Circuit depth
    quantum_shots=1024,           # Measurement shots

    # Agent parameters
    state_dim=20,                  # State space dimension
    action_dim=64,                 # Action space dimension
    learning_rate=1e-3,           # Learning rate
    gamma=0.99,                   # Discount factor

    # Training parameters
    max_episodes=500,             # Maximum episodes
    max_steps_per_episode=1000,   # Max steps per episode
    epsilon_start=0.3,            # Initial exploration
    epsilon_decay=0.995,          # Exploration decay
    epsilon_min=0.01,             # Minimum exploration

    # Mission parameters
    difficulty_level="EXTREME",   # Environment difficulty
    crisis_injection_rate=0.2,    # Crisis event rate
    quantum_weight_adaptive=True, # Adaptive quantum weighting

    # Hardware parameters
    device="auto",                # 'cpu', 'cuda', or 'auto'
    quantum_backend="simulator"   # 'simulator' or 'hardware'
)
```

## 📈 Performance Tracking

The package includes comprehensive performance tracking:

```python
from ca19_modular.utils import PerformanceTracker

tracker = PerformanceTracker()

# During training
for episode in range(num_episodes):
    # ... training code ...
    tracker.update_episode(
        episode_reward=reward,
        episode_length=length,
        metrics={
            'quantum_fidelity': fidelity,
            'td_error': error,
            'dopamine_level': dopamine
        }
    )

# Analysis
stats = tracker.get_summary_stats()
tracker.plot_training_progress('training_progress.png')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on research from CA19: Next-Generation Unified RL Systems
- Implements algorithms from quantum computing, neuromorphic engineering, and RL literature
- Inspired by cutting-edge research in hybrid quantum-classical systems

## 📞 Support

For questions, issues, or contributions, please:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Contact the maintainers

---

**Happy researching with CA19 Modular! 🚀**
