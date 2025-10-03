# Homework 5: Exploration, SAC, and Meta-Learning

This homework implements advanced deep reinforcement learning algorithms focusing on exploration methods, Soft Actor-Critic (SAC), and meta-learning.

## 🚀 Quick ### Key Hyperparameters

### SAC

- `--hidden_sizes`: Network hidden layer sizes (default: [256, 256])
- `--learning_rate`: Learning rate (default: 3e-3)
- `--alpha`: Temperature parameter (default: 1.0)
- `--batch_size`: Training batch size (default: 256)
- `--discount`: Discount factor (default: 0.99)
- `--tau`: Soft update coefficient (default: 0.01)
- `--reparameterize`: Use reparameterization trick

### Exploration

- `--bonus_coeff`: Exploration bonus coefficient (default: 1.0)
- `--initial_rollouts`: Initial rollouts for density estimation (default: 10)

### Meta-Learning

- `--num_tasks`: Number of tasks for meta-training (default: 20)
- `--meta_steps`: Number of meta-training steps (default: 100)
- `--meta_batch_size`: Meta batch size (default: 4)
- `--adaptation_steps`: Adaptation steps per task (default: 5)
- `--meta_learning_rate`: Meta learning rate (default: 1e-3)

## 🔧 MuJoCo Setup

Some environments (HalfCheetah, SparseHalfCheetah) require MuJoCo:

1. **Download MuJoCo**: [mujoco.org](https://www.mujoco.org/)
2. **Install mujoco-py**: `pip install mujoco-py`
3. **macOS**: `brew install gcc --without-multilib`
4. **Linux**: Install GCC 6/7 and development libraries

If MuJoCo is not installed, the automation script will skip MuJoCo-dependent experiments.

## 📊 Results Organization

After running `./run_all_hw5.sh`, results are organized as:

```
results_hw5/
├── logs/                              # Training logs and raw data
│   ├── sac_Pendulum-v0_*/            # SAC on basic environments
│   ├── sac_HalfCheetah-v2_*/         # SAC on MuJoCo (if available)
│   ├── exploration_MountainCar-v0_*/ # Exploration experiments
│   └── meta_CartPole-v0_*/           # Meta-learning experiments
└── plots/                             # Performance plots
    └── *_performance.png
```

## 🧠 Algorithm Overview

### Soft Actor-Critic (SAC)

Maximum entropy RL algorithm that learns:
- **Actor**: Stochastic policy with entropy regularization
- **Critic**: Two Q-networks (for stability)
- **Value**: State value network
- **Automatic Temperature Tuning**: Adaptive entropy coefficient

### Exploration with Density Models

Intrinsic motivation via novelty detection:
- **Density Model**: Neural network estimates state visitation frequency
- **Exploration Bonus**: Reward novelty (low-density states)
- **Bonus Coefficient**: Balances exploration vs exploitation

### Meta-Learning (MAML)

Few-shot adaptation across task distributions:
- **Inner Loop**: Fast adaptation to new tasks
- **Outer Loop**: Meta-optimization across tasks
- **Meta-Batch**: Sample tasks for meta-updates
- **Adaptation Steps**: Gradient steps per task

## 🔄 TensorFlow 2.x Compatibility

This codebase uses TensorFlow 2.x with v1 compatibility mode:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

All agents properly manage TensorFlow sessions for training and inference.

## 🐛 Troubleshooting

### MuJoCo Installation Issues

**Error**: `Could not find GCC 6 or GCC 7 executable`
```bash
# macOS
brew install gcc --without-multilib

# Linux
sudo apt-get install gcc-7 g++-7
```

### TensorFlow Compatibility

Ensure compatible TensorFlow version:
```bash
pip install 'tensorflow>=2.8.0,<2.16.0'
```

### Memory Issues

Reduce batch sizes:
```bash
python run_hw5.py sac --batch_size 128
```

## 👤 Author

**Saeed Reza Zouashkiani**  
Student ID: 400206262

## 📄 License

See `LICENSE` file in repository root.## Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make automation script executable
chmod +x run_all_hw5.sh
```

**Optional - MuJoCo**: Some experiments require MuJoCo. See [MuJoCo Setup](#mujoco-setup) below.

### Run All Experiments (Automated)

```bash
# Run complete training pipeline (SAC, Exploration, Meta-Learning)
./run_all_hw5.sh
```

This will:
- ✅ Train SAC on continuous control tasks
- ✅ Test exploration with density models
- ✅ Train meta-learning agents for few-shot adaptation
- ✅ Generate performance plots
- ✅ Organize all results in `results_hw5/`

## 📁 Modular Structure

The homework has been completely restructured into a professional, modular codebase:

```
hw5/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── sac_agent.py           # SAC agent implementation
│   │   ├── exploration_agent.py   # Exploration methods
│   │   └── meta_agent.py          # Meta-learning agents
│   ├── models/
│   │   ├── __init__.py
│   │   └── base_networks.py       # Base network components
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── replay_buffer.py       # Experience replay buffers
│   │   ├── data_structures.py     # Data structures and containers
│   │   ├── logger.py              # Logging utilities
│   │   └── normalization.py       # Normalization utilities
│   └── environments/
│       ├── __init__.py
│       └── wrappers.py            # Environment wrappers
├── configs/
│   ├── sac_config.py              # SAC hyperparameters
│   ├── exploration_config.py      # Exploration hyperparameters
│   └── meta_config.py             # Meta-learning hyperparameters
├── scripts/
│   ├── train_sac.py               # SAC training script
│   ├── train_exploration.py       # Exploration training script
│   └── train_meta.py              # Meta-learning training script
├── data/                          # Experiment results and data
├── docs/                          # Documentation
├── run_hw5.py                     # Main training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Components

### Agents (`src/agents/`)

#### `sac_agent.py`

Contains the Soft Actor-Critic agent:

- **`SACAgent`**: Main SAC agent with actor, critic, and value networks
- Support for reparameterization trick and automatic entropy tuning

#### `exploration_agent.py`

Contains exploration methods:

- **`ExplorationAgent`**: Continuous exploration with density models
- **`DiscreteExplorationAgent`**: Discrete exploration for discrete action spaces
- **`DensityModel`**: Neural network density estimator for state visitation

#### `meta_agent.py`

Contains meta-learning algorithms:

- **`MetaLearningAgent`**: Base meta-learning agent
- **`MAMLAgent`**: Model-Agnostic Meta-Learning implementation

### Models (`src/models/`)

#### `base_networks.py`

Base network components:

- **`ActorNetwork`**: Policy network for SAC
- **`CriticNetwork`**: Value network for SAC
- **`ValueNetwork`**: State value network for SAC

### Utilities (`src/utils/`)

#### `replay_buffer.py`

Experience replay buffers:

- **`ReplayBuffer`**: Standard experience replay
- **`PrioritizedReplayBuffer`**: Prioritized experience replay
- **`TrajectoryBuffer`**: Trajectory storage for meta-learning

#### `data_structures.py`

Data structures and containers:

- **`Dataset`**: Generic dataset class
- **`RollingDataset`**: Rolling window dataset
- **`Transition`**: Named tuple for transitions
- **`Trajectory`**: Named tuple for trajectories

#### `logger.py`

Logging utilities:

- **`Logger`**: Basic logging functionality
- **`WandBLogger`**: Weights & Biases integration

#### `normalization.py`

Normalization utilities:

- **`Normalizer`**: Base normalization class
- **`RewardNormalizer`**: Reward normalization
- **`StateNormalizer`**: State normalization
- **`ActionNormalizer`**: Action normalization

### Environments (`src/environments/`)

#### `wrappers.py`

Environment wrappers for preprocessing:

- **`TimeLimitWrapper`**: Episode time limits
- **`ActionRepeatWrapper`**: Action repetition
- **`FrameStackWrapper`**: Frame stacking
- **`RewardScaleWrapper`**: Reward scaling
- **`GrayscaleWrapper`**: Grayscale conversion
- **`ResizeWrapper`**: Image resizing

### Configurations (`configs/`)

Separate configuration files for each algorithm:

- **`sac_config.py`**: SAC hyperparameters
- **`exploration_config.py`**: Exploration hyperparameters
- **`meta_config.py`**: Meta-learning hyperparameters

### Scripts (`scripts/`)

Convenient training scripts:

- **`train_sac.py`**: SAC training with default config
- **`train_exploration.py`**: Exploration training with default config
- **`train_meta.py`**: Meta-learning training with default config

## Usage

### Quick Start with Scripts

Train with default configurations:

```bash
# SAC training
python scripts/train_sac.py

# Exploration training
python scripts/train_exploration.py

# Meta-learning training
python scripts/train_meta.py
```

### Custom Training with Main Script

#### Soft Actor-Critic

Train SAC agent:

```bash
python run_hw5.py sac --env_name Pendulum-v0 --total_steps 100000
```

#### Exploration Methods

Train exploration agent:

```bash
python run_hw5.py exploration --env_name MountainCar-v0 --bonus_coeff 0.1
```

#### Meta-Learning

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
