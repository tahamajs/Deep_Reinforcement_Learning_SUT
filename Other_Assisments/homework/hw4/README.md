# Homework 4: Model-Based Reinforcement Learning

This homework implements model-based reinforcement learning algorithms, focusing on learning dynamics models and using them for planning and control.

## 🚀 Quick Start

### Prerequisites

**Note**: This homework works with or without MuJoCo!

- **With MuJoCo** (recommended): Full experiments on HalfCheetah-v2
- **Without MuJoCo**: Fallback experiments on Pendulum-v0 (simpler but functional)

#### Option A: Run Without MuJoCo (Pendulum)

```bash
# Install dependencies
pip install -r requirements.txt

# Make script executable
chmod +x run_all_hw4.sh

# Run all experiments (automatically uses Pendulum if no MuJoCo)
./run_all_hw4.sh
```

#### Option B: Install MuJoCo for Full Experience

1. **Install MuJoCo**: Download from [mujoco.org](https://www.mujoco.org/) or [GitHub releases](https://github.com/deepmind/mujoco/releases)
2. **Extract to** `~/.mujoco/mujoco210`
3. **Install mujoco-py**: `pip install mujoco-py`
4. **macOS users**: Install GCC toolchain:
   ```bash
   brew install gcc
   ```
5. **Linux users**: Install GCC 6/7 and dev libraries:
   ```bash
   sudo apt-get install gcc g++ libgl1-mesa-dev libglew-dev
   ```

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make automation script executable
chmod +x run_all_hw4.sh run_all.sh
```

### Run All Experiments (Automated)

```bash
# Run complete training pipeline
./run_all_hw4.sh
# or
./run_all.sh

# Environment automatically selected:
# - HalfCheetah-v2 if MuJoCo available
# - Pendulum-v0 otherwise
```

This will:
- ✅ Detect MuJoCo availability
- ✅ Train dynamics models (Q1)
- ✅ Evaluate MPC policies (Q2)
- ✅ Run on-policy MBRL with various hyperparameters (Q3)
- ✅ Generate comparison plots
- ✅ Organize all results in `data/` and `plots/`

## 📁 Project Structure

```
hw4/
├── src/
│   └── model_based_rl.py          # Modular MBRL components (Dataset, DynamicsModel, MPCPolicy, Agent)
├── run_mbrl.py                    # Modular training script with TF2 compatibility
├── main.py                        # Original interface (updated with TF2 support)
├── half_cheetah_env.py           # HalfCheetah environment wrapper
├── logger.py                     # Logging utilities
├── plot.py                       # Plotting utilities
├── timer.py                      # Timing utilities
├── utils.py                      # General utilities
├── run_all_hw4.sh                # 🆕 Comprehensive automation script
├── requirements.txt              # Python dependencies (TF2-compatible)
└── README.md                     # This file
```

## 🎯 Individual Questions

### Question 1: Dynamics Model Training

Train a dynamics model on random data and evaluate predictions:

```bash
python run_mbrl.py q1 --exp_name my_experiment_q1
```

**What it does**:
- Collects random rollouts
- Trains neural network dynamics model
- Generates prediction plots (actual vs predicted states)

**Results**: Saved to `data/<exp_name>/prediction_rollout_*.jpg`

### Question 2: MPC with Random Shooting

Train dynamics model and evaluate MPC policy:

```bash
python run_mbrl.py q2 --exp_name my_experiment_q2 --render
```

**What it does**:
- Trains dynamics model on random data
- Uses Model Predictive Control (MPC) with random shooting for planning
- Evaluates policy performance

**Results**: Console output with return statistics

### Question 3: On-policy MBRL

Run iterative on-policy model-based RL:

```bash
# Default configuration
python main.py q3 --exp_name default

# Vary random action samples
python main.py q3 --exp_name action4096 --num_random_action_selection 4096

# Vary MPC horizon
python main.py q3 --exp_name horizon15 --mpc_horizon 15

# Vary NN layers
python main.py q3 --exp_name layers2 --nn_layers 2
```

**What it does**:
- Iteratively trains dynamics model and collects on-policy data
- Improves policy over multiple iterations
- Logs performance metrics

**Results**: Training curves saved to `data/<exp_name>/log.csv`

## 📊 Plotting

Generate comparison plots:

```bash
# Single experiment
python plot.py --exps HalfCheetah_q3_default --save my_plot

# Compare multiple experiments
python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 HalfCheetah_q3_action16384 --save action_comparison
```

**Results**: Plots saved to `plots/*.jpg`

## 🎛️ Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--nn_layers` | 1 | Number of hidden layers in dynamics model |
| `--training_epochs` | 60 | Training epochs for dynamics model |
| `--mpc_horizon` | 15 | MPC planning horizon |
| `--num_random_action_selection` | 4096 | Random actions sampled for MPC |
| `--num_init_random_rollouts` | 10 | Initial random rollouts |
| `--num_onpolicy_iters` | 10 | On-policy iterations (Q3) |
| `--num_onpolicy_rollouts` | 10 | Rollouts per on-policy iteration |
| `--max_rollout_length` | 500 | Maximum episode length |
| `--training_batch_size` | 512 | Batch size for dynamics model training |

## 🔧 Troubleshooting

### MuJoCo Installation Issues

**Error**: `Could not find GCC 6 or GCC 7 executable`
```bash
# macOS
brew install gcc --without-multilib

# Linux (Ubuntu/Debian)
sudo apt-get install gcc-7 g++-7
```

**Error**: `mujoco_py not found`
```bash
pip install mujoco-py
# May require: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

### TensorFlow Compatibility

This codebase uses TensorFlow 2.x with v1 compatibility mode (`tf.compat.v1`). If you encounter TF-related errors:

```bash
# Ensure compatible TensorFlow version
pip install 'tensorflow>=2.8.0,<2.16.0'
```

### Memory Issues

For large-scale experiments, reduce batch sizes or horizon:

```bash
python run_mbrl.py q3 --training_batch_size 256 --mpc_horizon 10
```

## 📈 Results Organization

After running `./run_all_hw4.sh`, results are organized as:

```
results_hw4/
├── logs/                           # Training logs and raw data
│   ├── q1_dynamics_*/              # Q1 results with prediction plots
│   ├── q2_mpc_*/                   # Q2 results
│   ├── HalfCheetah_q3_default/     # Q3 default run
│   ├── HalfCheetah_q3_action*/     # Q3 action sweep
│   ├── HalfCheetah_q3_horizon*/    # Q3 horizon sweep
│   └── HalfCheetah_q3_layers*/     # Q3 layers sweep
└── plots/                          # Comparison plots
    ├── HalfCheetah_q3_default.jpg
    ├── HalfCheetah_q3_actions.jpg
    ├── HalfCheetah_q3_mpc_horizon.jpg
    └── HalfCheetah_q3_nn_layers.jpg
```

## 🧠 Algorithm Overview

1. **Dynamics Learning**: Train neural network to predict next states from current state-action pairs
   - Input: `[state, action]`
   - Output: `delta_state` (change in state)
   - Loss: MSE between predicted and actual state changes

2. **MPC Planning**: Use learned dynamics for multi-step ahead planning
   - Sample random action sequences
   - Simulate trajectories using dynamics model
   - Select action sequence with highest predicted return
   - Execute first action, replan at next step

3. **On-policy Improvement**: Iteratively improve policy and dynamics model
   - Train dynamics model on collected data
   - Use MPC policy to collect new on-policy data
   - Aggregate data and retrain dynamics model
   - Repeat for multiple iterations

## 🔄 Updating from Original Code

This repository includes TensorFlow 2.x compatibility updates:

- ✅ `tf.compat.v1` imports with fallbacks
- ✅ Session management with proper initialization
- ✅ MuJoCo dependency checks
- ✅ Comprehensive automation scripts
- ✅ Non-interactive plotting (`matplotlib.use('Agg')`)

## 📝 Dependencies

- **TensorFlow**: 2.8.0 - 2.15.x (with v1 compatibility)
- **NumPy**: >=1.19.0
- **Gym**: 0.21.0 - 0.25.x
- **MuJoCo**: mujoco-py >=2.1.0 (optional, requires system setup)
- **Matplotlib**: >=3.5.0
- **Pandas, Scipy, Colorlog**: For logging and utilities

## 👤 Author

**Saeed Reza Zouashkiani**  
Student ID: 400206262

## 📄 License

See `LICENSE` file in repository root.
