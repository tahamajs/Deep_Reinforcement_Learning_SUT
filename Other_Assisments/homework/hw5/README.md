# Homework 5: Exploration and Meta-Learning

This homework implements advanced deep reinforcement learning algorithms focusing on exploration methods, Soft Actor-Critic (SAC), and meta-learning.

## Modular Structure

The homework has been restructured into modular components:

```
hw5/
├── src/
│   ├── sac_agent.py           # SAC agent with actor-critic networks
│   ├── exploration_agent.py   # Exploration methods with density models
│   └── meta_agent.py          # Meta-learning agents (MAML, etc.)
├── run_hw5.py                 # Modular training script
├── sac/                       # Original SAC implementation
├── exp/                       # Original exploration implementation
├── meta/                      # Original meta-learning implementation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Components

### `src/sac_agent.py`

Contains the Soft Actor-Critic agent:

- **`SACAgent`**: Main SAC agent with actor, critic, and value networks
- **`ReplayBuffer`**: Experience replay buffer
- Support for reparameterization trick and automatic entropy tuning

### `src/exploration_agent.py`

Contains exploration methods:

- **`ExplorationAgent`**: Continuous exploration with density models
- **`DiscreteExplorationAgent`**: Discrete exploration for discrete action spaces
- **`DensityModel`**: Neural network density estimator for state visitation

### `src/meta_agent.py`

Contains meta-learning algorithms:

- **`MetaLearningAgent`**: Base meta-learning agent
- **`MAMLAgent`**: Model-Agnostic Meta-Learning implementation
- **`ReplayBuffer`**: Task trajectory buffer

## Usage

### Soft Actor-Critic

Train SAC agent:

```bash
python run_hw5.py sac --env_name Pendulum-v0 --total_steps 100000
```

### Exploration Methods

Train exploration agent:

```bash
python run_hw5.py exploration --env_name MountainCar-v0 --bonus_coeff 0.1
```

### Meta-Learning

Train meta-learning agent:

```bash
python run_hw5.py meta --env_name CartPole-v0 --num_tasks 20 --meta_steps 100
```

## Key Hyperparameters

### SAC

- `--hidden_sizes`: Network hidden layer sizes (default: [256, 256])
- `--learning_rate`: Learning rate (default: 3e-3)
- `--alpha`: Temperature parameter (default: 1.0)
- `--batch_size`: Training batch size (default: 256)
- `--tau`: Soft update coefficient (default: 0.01)

### Exploration

- `--bonus_coeff`: Exploration bonus coefficient (default: 1.0)
- `--initial_rollouts`: Initial data collection rollouts (default: 10)

### Meta-Learning

- `--meta_learning_rate`: Meta learning rate (default: 1e-3)
- `--adaptation_steps`: Inner loop adaptation steps (default: 5)
- `--meta_batch_size`: Meta batch size (default: 4)
- `--num_tasks`: Number of tasks for training (default: 20)

## Algorithms Overview

### Soft Actor-Critic (SAC)

- **Off-policy**: Learns from replay buffer
- **Maximum entropy**: Encourages exploration through entropy bonus
- **Actor-critic**: Separate policy and value networks
- **Two Q-functions**: Addresses overestimation bias

### Exploration Methods

- **Density modeling**: Learns state visitation density
- **Reward bonuses**: Negative log density as exploration bonus
- **Count-based**: Intrinsic motivation for less visited states

### Meta-Learning

- **Few-shot adaptation**: Learns to adapt quickly to new tasks
- **MAML**: Model-Agnostic Meta-Learning algorithm
- **Task distribution**: Learns across multiple related tasks

## Results

Results are saved in the `data/` directory with timestamps. Each experiment includes:

- Training curves and metrics
- Model checkpoints (if applicable)
- Configuration parameters

## Dependencies

- tensorflow: Neural network framework
- numpy: Numerical computations
- gym: Reinforcement learning environments
- matplotlib: Plotting (optional)

## Installation

```bash
pip install -r requirements.txt
```

## Author

Saeed Reza Zouashkiani
Student ID: 400206262
