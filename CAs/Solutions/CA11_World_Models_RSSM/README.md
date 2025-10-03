# CA11: Advanced Model-Based Reinforcement Learning

This project implements advanced model-based reinforcement learning algorithms, including World Models, Recurrent State Space Models (RSSM), and Dreamer agents. The code has been modularized from the original notebook for better organization and reusability.

## Project Structure

```
CA11/
├── world_models/          # World model components
│   ├── __init__.py
│   ├── vae.py            # Variational Autoencoder
│   ├── dynamics.py       # Dynamics models
│   ├── reward_model.py   # Reward prediction models
│   ├── world_model.py    # Complete world model
│   ├── rssm.py          # Recurrent State Space Model
│   └── trainers.py      # Training utilities
├── agents/               # RL agents
│   ├── __init__.py
│   ├── latent_actor.py  # Actor for latent space
│   ├── latent_critic.py # Critic for latent space
│   └── dreamer_agent.py # Dreamer agent implementation
├── environments/         # Custom environments
│   ├── __init__.py
│   ├── continuous_cartpole.py
│   ├── continuous_pendulum.py
│   └── sequence_environment.py
├── utils/                # Utilities
│   ├── __init__.py
│   ├── data_collection.py
│   └── visualization.py
├── experiments/          # Experiment scripts
│   ├── world_model_experiment.py
│   ├── rssm_experiment.py
│   └── dreamer_experiment.py
├── CA11.ipynb           # Original notebook (updated to use modular code)
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Key Components

### World Models

- **VAE (Variational Autoencoder)**: Learns latent representations of observations
- **Dynamics Models**: Predict next latent states (stochastic and deterministic)
- **Reward Models**: Predict rewards in latent space
- **RSSM (Recurrent State Space Model)**: Temporal world modeling with recurrent networks

### Agents

- **Latent Actor**: Policy network for continuous actions in latent space
- **Latent Critic**: Value function for latent space
- **Dreamer Agent**: Complete model-based RL agent using imagination

### Environments

- **Continuous CartPole**: Continuous version of classic CartPole
- **Continuous Pendulum**: Continuous pendulum swing-up task
- **Sequence Environment**: Memory-requiring task for testing RSSM

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

1. **World Model Training**:

```bash
cd experiments
python world_model_experiment.py
```

2. **RSSM Training**:

```bash
python rssm_experiment.py
```

3. **Dreamer Agent Training**:

```bash
python dreamer_experiment.py
```

### Using Individual Components

```python
from world_models.vae import VariationalAutoencoder
from environments.continuous_cartpole import ContinuousCartPole
from utils.data_collection import collect_world_model_data

# Create environment and collect data
env = ContinuousCartPole()
data = collect_world_model_data(env, steps=1000, episodes=10)

# Create and train VAE
vae = VariationalAutoencoder(obs_dim=4, latent_dim=32, hidden_dims=[128, 64])
# ... training code ...
```

## Key Features

- **Modular Design**: Clean separation of world models, agents, environments, and utilities
- **Comprehensive Testing**: Each component can be tested independently
- **Visualization Tools**: Built-in plotting functions for analysis
- **Experiment Scripts**: Ready-to-run experiments for all major components
- **Extensible Architecture**: Easy to add new environments, models, or agents

## Algorithms Implemented

1. **World Models**: Learning predictive models of environments
2. **RSSM**: Recurrent state space models for temporal dependencies
3. **Dreamer**: Model-based RL using learned world models and imagination

## Dependencies

- PyTorch: Neural network framework
- NumPy: Numerical computations
- Matplotlib/Seaborn: Visualization
- tqdm: Progress bars
- Gymnasium: Reinforcement learning environments

## Notes

- All components are designed to work with continuous action spaces
- Models use PyTorch's GPU acceleration when available
- Visualization functions provide comprehensive analysis tools
- Experiment scripts include hyperparameter configurations and evaluation metrics
