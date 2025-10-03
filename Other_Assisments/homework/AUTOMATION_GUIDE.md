# Deep RL Homework Automation Guide

Complete automation setup for all homework assignments with TensorFlow 2.x compatibility, session management, and comprehensive result collection.

## 📋 Overview

All homework assignments now include:
- ✅ **TensorFlow 2.x Compatibility**: `tf.compat.v1` with proper session management
- ✅ **MuJoCo Guards**: Graceful handling of missing MuJoCo dependencies
- ✅ **Automation Scripts**: One-command execution for complete experiments
- ✅ **Result Organization**: Structured logs, plots, and videos
- ✅ **Error Handling**: Session-safe variable pickling and action formatting

## 🚀 Quick Start

### HW2: Policy Gradients

```bash
cd homework/hw2
chmod +x run_all_hw2.sh
./run_all_hw2.sh
```

**What it does**:
- Trains policy gradient agents on CartPole-v0, InvertedPendulum-v2, HalfCheetah-v2
- Records before/after training videos
- Generates learning curve plots
- Compares reward-to-go and baseline ablations

**Results**: `results_hw2/`

### HW3: DQN and Actor-Critic

```bash
cd homework/hw3
chmod +x run_all_hw3.sh
./run_all_hw3.sh
```

**What it does**:
- Trains DQN on LunarLander-v2
- Trains Actor-Critic on CartPole-v0, InvertedPendulum-v2
- Generates performance plots
- Organizes all logs and videos

**Results**: `results_hw3/`

### HW4: Model-Based RL

```bash
cd homework/hw4
chmod +x run_all_hw4.sh
./run_all_hw4.sh
```

**What it does**:
- Q1: Dynamics model training and prediction evaluation
- Q2: MPC with random shooting
- Q3: On-policy MBRL with hyperparameter sweeps
- Generates comparison plots

**Results**: `results_hw4/`

**Note**: Requires MuJoCo installation (see [MuJoCo Setup](#mujoco-setup))

## 🔧 Environment Setup

### Standard Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install common dependencies
pip install numpy matplotlib seaborn pandas gym tensorflow
```

### MuJoCo Setup

Required for HW2 (some envs), HW3 (some envs), and HW4 (all envs):

1. **Download MuJoCo**: [mujoco.org](https://www.mujoco.org/)
   - Extract to `~/.mujoco/mujoco210/`

2. **Install mujoco-py**:
   ```bash
   pip install mujoco-py
   ```

3. **macOS - Install GCC**:
   ```bash
   brew install gcc --without-multilib
   ```

4. **Linux - Install GCC and libraries**:
   ```bash
   sudo apt-get install gcc-7 g++-7
   sudo apt-get install libgl1-mesa-dev libglew-dev libosmesa6-dev
   ```

5. **Verify Installation**:
   ```bash
   python -c "import mujoco_py"
   ```

If MuJoCo is not installed, automation scripts will skip MuJoCo-dependent environments and continue with available ones.

## 📊 Results Structure

Each homework creates an organized results directory:

```
results_hw{2,3,4}/
├── logs/                    # Training logs and raw data
│   ├── {experiment_name}/
│   │   ├── log.txt         # Training metrics
│   │   ├── log.csv         # CSV format for plotting
│   │   └── vars.pkl        # Model checkpoints
├── plots/                   # Learning curves and comparisons
│   └── *.png
└── videos/                  # Before/after training videos (HW2, HW3)
    ├── before_*.mp4
    └── after_*.mp4
```

## 🐛 Common Issues and Solutions

### Issue: "No default session is registered"

**Fixed** ✅ - All code now uses explicit session passing:
```python
logz.pickle_tf_vars(agent.sess)  # Pass session explicitly
```

### Issue: "AssertionError: array([1]) (<class 'numpy.ndarray'>) invalid"

**Fixed** ✅ - Discrete actions now converted to Python ints:
```python
if self.discrete:
    ac = int(np.asarray(ac).reshape(-1)[0])
```

### Issue: MuJoCo GCC error

**Solution**: Install compatible GCC toolchain:
```bash
# macOS
brew install gcc --without-multilib

# Linux
sudo apt-get install gcc-7 g++-7
```

### Issue: TensorFlow compatibility warnings

**Normal** ⚠️ - Using `tf.compat.v1` for backward compatibility:
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

### Issue: Gym rendering on headless servers

**Solution**: Use `Agg` backend for matplotlib:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

## 🎯 Individual Experiment Examples

### HW2: Policy Gradient Variants

```bash
cd homework/hw2

# Vanilla policy gradient
python run_pg.py CartPole-v0 --exp_name vpg -n 100 -b 1000

# With reward-to-go
python run_pg.py CartPole-v0 --exp_name vpg_rtg -n 100 -b 1000 --reward_to_go

# With baseline
python run_pg.py CartPole-v0 --exp_name vpg_baseline -n 100 -b 1000 --nn_baseline

# With reward-to-go + baseline
python run_pg.py CartPole-v0 --exp_name vpg_rtg_baseline -n 100 -b 1000 --reward_to_go --nn_baseline

# Record videos
python run_pg.py CartPole-v0 --exp_name vpg_video -n 100 -b 1000 --record_video
```

### HW3: DQN and Actor-Critic

```bash
cd homework/hw3

# DQN on LunarLander
python run_dqn_lander.py LunarLander-v2 --num_timesteps 50000 --seed 1

# Actor-Critic on CartPole
python run_ac.py CartPole-v0 --exp_name ac_cartpole -n 100 -b 1000 -lr 0.005

# Actor-Critic on InvertedPendulum (requires MuJoCo)
python run_ac.py InvertedPendulum-v2 --exp_name ac_pendulum -n 150 -b 5000 -lr 0.005
```

### HW4: Model-Based RL

```bash
cd homework/hw4

# Q1: Train dynamics model
python run_mbrl.py q1 --exp_name my_q1_run --training_epochs 60

# Q2: Evaluate MPC policy
python run_mbrl.py q2 --exp_name my_q2_run --mpc_horizon 15 --render

# Q3: On-policy MBRL
python main.py q3 --exp_name my_q3_run --num_onpolicy_iters 10

# Q3 with custom hyperparameters
python main.py q3 --exp_name custom \
    --num_random_action_selection 4096 \
    --mpc_horizon 20 \
    --nn_layers 2
```

## 📈 Plotting Results

### HW2: Compare Policy Gradient Variants

```bash
cd homework/hw2
python plot.py \
    --data_dir data \
    --env_name CartPole-v0 \
    --exp_names vpg vpg_rtg vpg_baseline vpg_rtg_baseline \
    --output plots/pg_comparison.png
```

### HW3: Plot Actor-Critic Learning Curves

Already generated by automation script in `results_hw3/plots/`

### HW4: Compare Hyperparameters

```bash
cd homework/hw4
python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 --save action_comparison
```

## 🔄 Automation Script Architecture

All automation scripts follow a consistent pattern:

1. **Environment Setup**: Create result directories, check dependencies
2. **MuJoCo Detection**: Test for `import mujoco_py`, skip or exit gracefully
3. **Training Phase**: Run experiments with proper logging
4. **Log Collection**: Move experiment logs to organized structure
5. **Plotting Phase**: Generate comparison plots using collected data
6. **Summary**: Print results locations and re-run commands

### Script Flow

```bash
./run_all_hw{2,3,4}.sh
    ↓
Check MuJoCo availability
    ↓
Run experiments (skip MuJoCo envs if unavailable)
    ↓
Collect logs to results_hw{2,3,4}/logs/
    ↓
Generate plots to results_hw{2,3,4}/plots/
    ↓
Record videos to results_hw{2,3,4}/videos/ (HW2, HW3)
    ↓
Print summary and instructions
```

## 🧪 Testing Your Setup

### Test TensorFlow Installation

```bash
python -c "import tensorflow.compat.v1 as tf; tf.disable_v2_behavior(); print('TF OK')"
```

### Test Gym Environments

```bash
# Test basic Gym
python -c "import gym; env = gym.make('CartPole-v0'); print('Gym OK')"

# Test MuJoCo environments
python -c "import gym; env = gym.make('InvertedPendulum-v2'); print('MuJoCo OK')"
```

### Run Quick Smoke Tests

```bash
# HW2: 5 iterations
cd homework/hw2
python run_pg.py CartPole-v0 --exp_name smoke_test -n 5 -b 100

# HW3: 5 iterations
cd homework/hw3
python run_ac.py CartPole-v0 --exp_name smoke_test -n 5 -b 100

# HW4: Q1 with minimal epochs
cd homework/hw4
python run_mbrl.py q1 --exp_name smoke_test --training_epochs 5 --num_init_random_rollouts 2
```

## 📚 Documentation

Each homework directory includes:
- `README.md`: Detailed usage, hyperparameters, and algorithm descriptions
- `requirements.txt`: Python dependencies with compatible versions
- `run_all_hw*.sh`: Comprehensive automation script
- Individual runner scripts with argument parsing

## 🎓 Key Improvements Made

### TensorFlow 2.x Compatibility
- Replaced `import tensorflow as tf` → `import tensorflow.compat.v1 as tf`
- Added `tf.disable_v2_behavior()` at module level
- Explicit session management with `sess.__enter__()`

### Session Management
- Updated `logz.pickle_tf_vars()` to accept explicit session parameter
- All callers pass `agent.sess` instead of relying on default session
- Proper session initialization with `tf.ConfigProto`

### Action Handling
- Discrete actions converted to Python `int` before `env.step()`
- Proper dtype handling in rollout buffers (`np.int32` vs `np.float32`)

### MuJoCo Environment Guards
- Centralized `MUJOCO_ENVS` sets in each homework
- Graceful skip with installation instructions
- No hard crashes when MuJoCo unavailable

### Automation Enhancements
- Absolute path resolution (`SCRIPT_DIR`)
- Proper log directory detection and movement
- Parallel-safe experiment naming with timestamps
- Comprehensive error messages and next-step guidance

## 🤝 Contributing

When adding new experiments:
1. Add TF v1 compatibility imports
2. Use explicit session passing
3. Add MuJoCo environment to guard list if applicable
4. Update automation script with new experiment
5. Document in README

## 📞 Support

If you encounter issues:
1. Check [Common Issues](#common-issues-and-solutions)
2. Verify environment setup with [Testing Your Setup](#testing-your-setup)
3. Review individual homework READMEs
4. Check experiment logs in `results_hw*/logs/*/log.txt`

## 🏆 Credits

**Author**: Saeed Reza Zouashkiani  
**Student ID**: 400206262  
**Course**: Deep Reinforcement Learning  
**Institution**: SUT (Sharif University of Technology)

---

**Last Updated**: October 3, 2025  
**TensorFlow Version**: 2.8.0 - 2.15.x (with v1 compatibility)  
**Python Version**: 3.7+
