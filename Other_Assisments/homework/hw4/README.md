# Homework 4: Model-Based Reinforcement Learning

This homework implements model-based reinforcement learning algorithms, focusing on learning dynamics models and using them for planning and control.

## Modular Structure

The homework has been restructured into modular components for better organization and maintainability:

```
hw4/
├── src/
│   └── model_based_rl.py          # Main MBRL agent with dynamics model and MPC
├── run_mbrl.py                    # Modular training script with argument parsing
├── half_cheetah_env.py           # HalfCheetah environment wrapper
├── logger.py                     # Logging utilities
├── plot.py                       # Plotting utilities
├── timer.py                      # Timing utilities
├── tabulate.py                   # Tabulation utilities
├── utils.py                      # General utilities (Dataset, RandomPolicy, etc.)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Components

### `src/model_based_rl.py`

Contains the main modular components:

- **`Dataset`**: Manages transition data storage and iteration
- **`DynamicsModel`**: Neural network for learning environment dynamics
- **`MPCPolicy`**: Model Predictive Control policy using random shooting
- **`ModelBasedRLAgent`**: Main agent class coordinating all components
- **`RandomPolicy`**: Random action policy for data collection

### `run_mbrl.py`

Modular training script that supports:

- **Q1**: Dynamics model training and prediction evaluation
- **Q2**: MPC policy evaluation
- **Q3**: On-policy MBRL with iterative improvement

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Question 1: Dynamics Model Training

Train a dynamics model on random data and evaluate predictions:

```bash
python run_mbrl.py q1 --exp_name my_experiment_q1
```

### Question 2: MPC with Random Shooting

Train dynamics model and evaluate MPC policy:

```bash
python run_mbrl.py q2 --exp_name my_experiment_q2 --render
```

### Question 3: On-policy MBRL

Run iterative on-policy model-based RL:

```bash
python run_mbrl.py q3 --exp_name my_experiment_q3 --num_onpolicy_iters 10
```

## Key Hyperparameters

- `--nn_layers`: Number of hidden layers in dynamics model (default: 1)
- `--training_epochs`: Training epochs for dynamics model (default: 60)
- `--mpc_horizon`: MPC planning horizon (default: 15)
- `--num_random_action_selection`: Random actions for MPC (default: 4096)
- `--num_init_random_rollouts`: Initial random rollouts (default: 10)
- `--num_onpolicy_iters`: On-policy iterations for Q3 (default: 10)

## Results

Results are saved in the `data/` directory with experiment names. Each experiment includes:

- Training logs (`log.txt`)
- Prediction plots for Q1 (`prediction_rollout_XXX.jpg`)
- Performance metrics logged to console

## Algorithm Overview

1. **Dynamics Learning**: Train neural network to predict next states from current state-action pairs
2. **MPC Planning**: Use learned dynamics for multi-step ahead planning with random action sampling
3. **On-policy Improvement**: Iteratively collect data with current policy and retrain dynamics model

## Dependencies

- tensorflow: Neural network framework
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- gym: Reinforcement learning environments

## Author

Saeed Reza Zouashkiani
Student ID: 400206262
