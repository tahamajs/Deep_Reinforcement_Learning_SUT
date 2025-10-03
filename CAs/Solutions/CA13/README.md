# CA13: Advanced Model-Based RL and World Models

This repository contains implementations for Computer Assignment 13 of the Deep Reinforcement Learning course, focusing on advanced model-based reinforcement learning techniques including world models, sample efficiency methods, and hierarchical RL.

## 📚 Overview

This assignment explores cutting-edge techniques in deep reinforcement learning:

- **Model-Based vs Model-Free RL**: Comparative analysis of approaches
- **World Models**: Variational Autoencoder-based environment modeling
- **Imagination-Based Learning**: Planning in latent space
- **Sample Efficiency**: Prioritized replay, data augmentation, auxiliary tasks
- **Transfer Learning**: Multi-task and meta-learning approaches
- **Hierarchical RL**: Options framework and feudal networks
- **Multi-Agent RL**: Coordination and communication protocols

## 🏗️ Project Structure

```
CA13/
├── CA13.ipynb              # Main assignment notebook
├── agents/                 # RL agent implementations
│   ├── __init__.py
│   ├── model_free.py      # DQN and other model-free agents
│   ├── model_based.py     # Model-based RL agents
│   ├── sample_efficient.py # Sample efficiency techniques
│   └── hierarchical.py    # Hierarchical RL agents
├── models/                 # Neural network architectures
│   ├── __init__.py
│   └── world_model.py     # VAE-based world models
├── environments/           # Custom environments
│   ├── __init__.py
│   └── grid_world.py      # Grid world environments
├── buffers/                # Experience replay buffers
│   ├── __init__.py
│   └── replay_buffer.py   # Standard and prioritized replay
├── evaluation/             # Evaluation framework
│   ├── __init__.py
│   └── advanced_evaluator.py # Comprehensive evaluation tools
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── visualization.py   # Plotting and analysis tools
│   └── helpers.py         # Helper functions
├── training_examples.py    # Training scripts and examples
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd CA13
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main notebook:

```bash
jupyter notebook CA13.ipynb
```

### Basic Usage

```python
from agents.model_free import DQNAgent
from agents.model_based import ModelBasedAgent
from models.world_model import VariationalWorldModel
from environments.grid_world import SimpleGridWorld
from training_examples import train_dqn_agent, evaluate_agent

# Create environment
env = SimpleGridWorld(size=5)

# Create agent
agent = DQNAgent(
    state_dim=2,
    action_dim=4,
    learning_rate=1e-3
)

# Train agent
results = train_dqn_agent(env, agent, num_episodes=200)

# Evaluate agent
eval_results = evaluate_agent(env, agent, num_episodes=10)
```

## 🔬 Key Components

### World Models

```python
from models.world_model import VariationalWorldModel

# Create VAE-based world model
world_model = VariationalWorldModel(
    obs_dim=4,
    action_dim=2,
    latent_dim=32
)

# Train on environment data
losses = world_model.compute_loss(obs, actions, rewards, next_obs, dones)

# Generate imagined trajectories
imagined_trajectory = world_model.imagine_trajectory(z_start, actions, horizon=10)
```

### Sample Efficient Agents

```python
from agents.sample_efficient import SampleEfficientAgent

# Create sample-efficient agent with prioritized replay and auxiliary tasks
agent = SampleEfficientAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=128
)

# Training includes data augmentation and auxiliary learning
losses = agent.update(batch_size=32)
```

### Hierarchical RL

```python
from agents.hierarchical import OptionsCriticAgent, FeudalAgent

# Options-Critic agent
oc_agent = OptionsCriticAgent(
    state_dim=4,
    action_dim=2,
    num_options=4
)

# Feudal Networks agent
feudal_agent = FeudalAgent(
    state_dim=4,
    action_dim=2,
    goal_dim=16,
    temporal_horizon=10
)
```

### Multi-Agent RL

```python
from environments.grid_world import MultiAgentGridWorld
from agents.model_free import MultiAgentDQN

# Create multi-agent environment
ma_env = MultiAgentGridWorld(size=7, n_agents=3)

# Create multi-agent system with communication
ma_system = MultiAgentDQN(
    n_agents=3,
    state_dim=obs_dim,
    action_dim=4,
    enable_communication=True
)
```

## 📊 Evaluation Framework

The evaluation framework provides comprehensive analysis tools:

```python
from evaluation.advanced_evaluator import AdvancedRLEvaluator

# Create evaluator
evaluator = AdvancedRLEvaluator(
    environments=[env1, env2, env3],
    agents={'DQN': dqn_agent, 'MB': mb_agent},
    metrics=['sample_efficiency', 'reward', 'transfer']
)

# Run comprehensive evaluation
results = evaluator.comprehensive_evaluation()
evaluator.generate_report()
evaluator.plot_results()
```

## 🎯 Experiments

### Model-Free vs Model-Based Comparison

Compare different RL approaches on the same environment:

```python
from training_examples import compare_agents

results = compare_agents(
    env=env,
    agents={
        'DQN': dqn_agent,
        'Model-Based': mb_agent,
        'Sample-Efficient': se_agent
    },
    num_episodes=200
)

# Plot comparison
plot_training_curves(results, save_path='comparison.png')
```

### Hyperparameter Sweep

```python
from training_examples import hyperparameter_sweep

param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'hidden_dim': [64, 128, 256],
    'gamma': [0.95, 0.99, 0.995]
}

results_df = hyperparameter_sweep(
    env=env,
    agent_class=DQNAgent,
    param_grid=param_grid,
    num_episodes=100
)
```

## 📈 Visualization Tools

The utils module provides extensive visualization capabilities:

```python
from utils.visualization import (
    plot_training_curves,
    plot_world_model_analysis,
    plot_multi_agent_analysis,
    create_summary_table
)

# Plot training progress
plot_training_curves({
    'DQN': dqn_rewards,
    'Model-Based': mb_rewards
}, title='Learning Curves Comparison')

# Analyze world model
plot_world_model_analysis(world_model_results)

# Multi-agent coordination analysis
plot_multi_agent_analysis(ma_results)

# Create summary table
summary_df = create_summary_table(results)
print(summary_df)
```

## 🔧 Configuration

Use the configuration system for reproducible experiments:

```python
from utils.helpers import create_config_template, save_config, load_config

# Create configuration
config = create_config_template()

# Modify configuration
config['agent']['learning_rate'] = 5e-4
config['training']['num_episodes'] = 500

# Save configuration
save_config(config, 'experiment_config.json')

# Load configuration
config = load_config('experiment_config.json')
```

## 📚 Theoretical Background

This implementation is based on several key papers:

- **World Models** (Ha & Schmidhuber, 2018): VAE-based world modeling
- **Dreamer** (Hafner et al., 2020): Imagination-based learning
- **Options-Critic** (Bacon et al., 2017): Hierarchical RL
- **Feudal Networks** (Vezhnevets et al., 2017): Manager-worker hierarchies
- **Prioritized Experience Replay** (Schaul et al., 2016): Sample efficiency

## 🎓 Learning Objectives

By completing this assignment, students will:

1. **Understand** the trade-offs between model-free and model-based RL
2. **Implement** world models using variational autoencoders
3. **Apply** imagination-based planning in latent space
4. **Design** sample-efficient learning algorithms
5. **Build** hierarchical decision-making systems
6. **Develop** multi-agent coordination protocols
7. **Evaluate** advanced RL methods comprehensively

## 🚀 Advanced Features

- **Modular Architecture**: Easy to extend and modify
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Multi-Agent Support**: Communication and coordination protocols
- **World Model Integration**: VAE-based environment modeling
- **Hierarchical Learning**: Options and feudal networks
- **Sample Efficiency**: Prioritized replay and auxiliary tasks
- **Transfer Learning**: Multi-task and meta-learning frameworks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Course instructors for guidance and feedback
- OpenAI Gym/Gymnasium for environment frameworks
- PyTorch team for deep learning framework
- Research community for foundational papers

---

**Happy Learning!** 🎉

For questions or issues, please open an issue in the repository.
