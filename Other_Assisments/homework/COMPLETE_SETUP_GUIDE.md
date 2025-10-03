# 🎓 Deep RL Homework Suite - Complete Setup

All homework assignments (HW2-HW5) have been upgraded with TensorFlow 2.x compatibility, comprehensive automation, and professional documentation.

## 📚 Homework Overview

| HW | Topic | Key Algorithms | MuJoCo Required |
|----|-------|----------------|-----------------|
| **HW2** | Policy Gradients | VPG, Reward-to-Go, Baseline | Partial |
| **HW3** | Value-Based Methods | DQN, Actor-Critic | Partial |
| **HW4** | Model-Based RL | Dynamics Models, MPC | **Yes** |
| **HW5** | Advanced Topics | SAC, Exploration, Meta-Learning | Partial |

## 🚀 Quick Start

### One Command for Everything

```bash
cd homework
chmod +x run_all_homeworks.sh
./run_all_homeworks.sh
```

This will:
1. Check MuJoCo availability
2. Run all homework experiments
3. Generate plots and videos
4. Organize results
5. Display summary

### Individual Homeworks

```bash
# HW2: Policy Gradients
cd hw2 && ./run_all_hw2.sh

# HW3: DQN & Actor-Critic
cd hw3 && ./run_all_hw3.sh

# HW4: Model-Based RL
cd hw4 && ./run_all_hw4.sh

# HW5: Exploration, SAC, Meta-Learning
cd hw5 && ./run_all_hw5.sh
```

### Selective Execution

```bash
./run_all_homeworks.sh --hw2-only  # Only HW2
./run_all_homeworks.sh --hw3-only  # Only HW3
./run_all_homeworks.sh --hw4-only  # Only HW4
./run_all_homeworks.sh --hw5-only  # Only HW5
```

## 🔧 Setup

### Basic Setup (All Homeworks)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy matplotlib seaborn pandas gym tensorflow
```

### MuJoCo Setup (Required for HW4, Optional for Others)

1. **Download MuJoCo**: [mujoco.org](https://www.mujoco.org/)
   - Extract to `~/.mujoco/mujoco210/`

2. **Install mujoco-py**:
   ```bash
   pip install mujoco-py
   ```

3. **Install GCC toolchain**:
   ```bash
   # macOS
   brew install gcc --without-multilib
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install gcc-7 g++-7
   sudo apt-get install libgl1-mesa-dev libglew-dev libosmesa6-dev
   ```

4. **Verify**:
   ```bash
   python -c "import mujoco_py"
   ```

## 📊 Results

Each homework creates organized results:

```
homework/
├── hw2/results_hw2/
│   ├── logs/       # Training logs
│   ├── plots/      # Learning curves
│   └── videos/     # Before/after videos
├── hw3/results_hw3/
│   ├── logs/
│   ├── plots/
│   └── videos/
├── hw4/results_hw4/
│   ├── logs/
│   └── plots/
└── hw5/results_hw5/
    ├── logs/
    └── plots/
```

## 🎯 Key Features

### 1. TensorFlow 2.x Compatibility

All code uses TF2 with v1 compatibility mode:
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

Proper session management throughout.

### 2. MuJoCo Environment Guards

Graceful handling of missing MuJoCo:
- Detects availability at runtime
- Skips MuJoCo experiments with helpful messages
- Continues with non-MuJoCo environments
- Provides installation instructions

### 3. Comprehensive Automation

Each homework has a `run_all_hw*.sh` script:
- Runs all experiments
- Organizes results
- Generates plots
- Records videos (HW2, HW3)
- Provides summary and instructions

### 4. Professional Documentation

- **Individual READMEs**: Complete usage guide per homework
- **AUTOMATION_GUIDE.md**: Master documentation
- **Migration summaries**: TF2 compatibility notes

### 5. Error Handling

Fixed common issues:
- Session management ("no default session")
- Action formatting (discrete action assertions)
- Video recording compatibility
- Plotting on headless servers

## 🧪 Testing

Quick smoke tests to verify setup:

```bash
# Test HW2 (5 iterations)
cd hw2
python run_pg.py CartPole-v0 --exp_name smoke_test -n 5 -b 100

# Test HW3 (5 iterations)
cd hw3
python run_ac.py CartPole-v0 --exp_name smoke_test -n 5 -b 100

# Test HW5 SAC (5k steps)
cd hw5
python run_hw5.py sac --env_name Pendulum-v0 --total_steps 5000 --exp_name smoke_test
```

## 📖 Documentation

- **AUTOMATION_GUIDE.md**: Comprehensive automation guide
- **hw2/README.md**: Policy gradients details
- **hw3/README.md**: DQN & Actor-Critic details
- **hw4/README.md**: Model-based RL details
- **hw5/README.md**: Exploration, SAC, meta-learning details

## 🐛 Common Issues

### "No default session is registered"

**Fixed** ✅ - All code now passes sessions explicitly:
```python
logz.pickle_tf_vars(agent.sess)
```

### "AssertionError: array([1]) invalid"

**Fixed** ✅ - Discrete actions converted to Python ints:
```python
if self.discrete:
    ac = int(np.asarray(ac).reshape(-1)[0])
```

### MuJoCo GCC Error

**Solution**: Install compatible GCC:
```bash
# macOS
brew install gcc --without-multilib

# Linux
sudo apt-get install gcc-7 g++-7
```

### Import Errors

**Solution**: Check Python path and reinstall:
```bash
pip install -e .  # If package structure
# or
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🎓 Course Information

**Course**: Deep Reinforcement Learning  
**Institution**: SUT (Sharif University of Technology)  
**Author**: Saeed Reza Zouashkiani  
**Student ID**: 400206262

## 🔄 System Requirements

- **Python**: 3.7+
- **TensorFlow**: 2.8.0 - 2.15.x (with v1 compatibility)
- **NumPy**: <2.0.0
- **Gym**: 0.21.0 - 0.25.x
- **MuJoCo**: mujoco-py >=2.1.0 (optional)

## 📝 Usage Examples

### HW2: Policy Gradient Variants

```bash
cd hw2

# Vanilla PG
python run_pg.py CartPole-v0 --exp_name vpg -n 100 -b 1000

# With reward-to-go
python run_pg.py CartPole-v0 --exp_name vpg_rtg -n 100 -b 1000 --reward_to_go

# With baseline
python run_pg.py CartPole-v0 --exp_name vpg_bl -n 100 -b 1000 --nn_baseline

# Full featured
python run_pg.py CartPole-v0 --exp_name vpg_full -n 100 -b 1000 \
    --reward_to_go --nn_baseline --record_video
```

### HW3: DQN and Actor-Critic

```bash
cd hw3

# DQN on LunarLander
python run_dqn_lander.py LunarLander-v2 --num_timesteps 50000

# Actor-Critic on CartPole
python run_ac.py CartPole-v0 --exp_name ac_cartpole -n 100 -b 1000
```

### HW4: Model-Based RL

```bash
cd hw4

# Q1: Dynamics model
python run_mbrl.py q1 --exp_name q1_run

# Q2: MPC
python run_mbrl.py q2 --exp_name q2_run

# Q3: On-policy MBRL
python main.py q3 --exp_name q3_run
```

### HW5: Advanced Algorithms

```bash
cd hw5

# SAC
python run_hw5.py sac --env_name Pendulum-v0 --total_steps 50000

# Exploration
python run_hw5.py exploration --env_name MountainCar-v0

# Meta-learning
python run_hw5.py meta --env_name CartPole-v0 --num_tasks 20
```

## 📈 Expected Results

### HW2
- CartPole: 200 average return (solved)
- InvertedPendulum: 1000+ average return
- Learning curves showing baseline improvement

### HW3
- LunarLander DQN: 200+ average return
- CartPole AC: 200 average return (solved)
- Before/after videos showing improvement

### HW4
- Q1: Accurate dynamics predictions (low MSE)
- Q2: Positive returns with MPC
- Q3: Improving returns over iterations

### HW5
- SAC: Smooth learning curves, exploration vs exploitation
- Exploration: Improved sample efficiency on sparse rewards
- Meta-learning: Fast adaptation to new tasks

## 🏆 Status

All homework assignments are:
- ✅ TensorFlow 2.x compatible
- ✅ Session-safe
- ✅ MuJoCo-aware
- ✅ Fully automated
- ✅ Professionally documented
- ✅ Production-ready

**Ready to run!** 🚀

---

**Last Updated**: October 3, 2025  
**Version**: 2.0 (TF2 Compatible)
