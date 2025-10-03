# MuJoCo Compatibility Guide

This document explains how all homework assignments (HW2-HW5) handle MuJoCo environments gracefully.

## Overview

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine used for robotics simulations. However, it requires additional setup and is not always available on all systems. All homework scripts have been updated to:

1. ✅ **Detect MuJoCo availability** at runtime
2. ✅ **Skip MuJoCo environments gracefully** if not installed
3. ✅ **Continue with non-MuJoCo environments** (CartPole, LunarLander, MountainCar, etc.)
4. ✅ **Provide clear installation instructions** when MuJoCo is needed

## MuJoCo Environments by Homework

### HW2: Policy Gradients
- **MuJoCo Environment**: HalfCheetah-v2
- **Non-MuJoCo Environments**: CartPole-v0, LunarLander-v2
- **Behavior**: If MuJoCo not available, trains on CartPole and LunarLander only

### HW3: Actor-Critic
- **MuJoCo Environment**: HalfCheetah-v2
- **Non-MuJoCo Environments**: CartPole-v0, InvertedPendulum-v2
- **Behavior**: If MuJoCo not available, trains on CartPole and InvertedPendulum only

### HW4: Model-Based RL
- **MuJoCo Environment**: HalfCheetah-v2
- **Non-MuJoCo Environment**: None (all experiments require MuJoCo)
- **Behavior**: If MuJoCo not available, exits with installation instructions

### HW5: Advanced RL
- **MuJoCo Environments**: HalfCheetah-v2, Ant-v2
- **Non-MuJoCo Environments**: Pendulum-v0, MountainCarContinuous-v0, CartPole-v0
- **Behavior**: If MuJoCo not available, trains on Pendulum, MountainCar, and CartPole only

## How It Works

### 1. Runtime Detection

Each homework script includes MuJoCo detection:

```python
# In Python scripts (e.g., run_pg.py, run_ac.py)
MUJOCO_ENVS = {
    'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'Ant-v2',
    'Humanoid-v2', 'Reacher-v2', 'Swimmer-v2', ...
}

def check_mujoco_available():
    try:
        import mujoco_py
        return True
    except (ImportError, Exception):
        return False
```

```bash
# In bash scripts (e.g., run_all.sh)
MUJOCO_AVAILABLE=false
python -c "import mujoco_py" 2>/dev/null && MUJOCO_AVAILABLE=true
```

### 2. Graceful Skipping

When a MuJoCo environment is requested but not available:

```python
if args.env_name in MUJOCO_ENVS and not check_mujoco_available():
    print(f"\n❌ ERROR: Environment '{args.env_name}' requires MuJoCo, but it's not installed.")
    print("\n📦 To install MuJoCo:")
    print("   1. Download MuJoCo 2.1.0 from: https://github.com/deepmind/mujoco/releases")
    print("   2. Extract to ~/.mujoco/mujoco210")
    print("   3. Install mujoco-py: pip install mujoco-py")
    print("   4. On macOS, you may need: brew install gcc")
    print("\nSkipping this experiment.\n")
    sys.exit(1)
```

### 3. Selective Environment Lists

Automation scripts adjust their environment lists based on MuJoCo availability:

```bash
if [ "$MUJOCO_AVAILABLE" = true ]; then
    ENVS=("CartPole-v0" "LunarLander-v2" "HalfCheetah-v2")
else
    ENVS=("CartPole-v0" "LunarLander-v2")
fi
```

## Installing MuJoCo

### macOS

```bash
# Download MuJoCo 2.1.0
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz

# Extract to ~/.mujoco
mkdir -p ~/.mujoco
tar -xzf mujoco210-macos-x86_64.tar.gz -C ~/.mujoco

# Install GCC (required for mujoco-py compilation)
brew install gcc

# Install mujoco-py
pip install mujoco-py

# Test installation
python -c "import mujoco_py; print('MuJoCo installed successfully!')"
```

### Ubuntu/Linux

```bash
# Download MuJoCo 2.1.0
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz

# Extract to ~/.mujoco
mkdir -p ~/.mujoco
tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco

# Install dependencies
sudo apt-get update
sudo apt-get install gcc g++ libgl1-mesa-dev libglew-dev patchelf

# Install mujoco-py
pip install mujoco-py

# Test installation
python -c "import mujoco_py; print('MuJoCo installed successfully!')"
```

## Common Issues

### Issue 1: GCC Not Found

**Error**: `Could not find GCC 6 or GCC 7 executable`

**Solution**:
```bash
# macOS
brew install gcc

# Ubuntu
sudo apt-get install gcc g++
```

### Issue 2: Missing OpenGL Libraries

**Error**: `Cannot load library libGL.so.1`

**Solution**:
```bash
# Ubuntu
sudo apt-get install libgl1-mesa-dev libglew-dev

# macOS (usually not needed)
brew install glew
```

### Issue 3: mujoco-py Build Fails

**Error**: Various compilation errors during `pip install mujoco-py`

**Solutions**:
1. Make sure MuJoCo is in `~/.mujoco/mujoco210`
2. Set environment variable:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
   ```
3. Try installing from source:
   ```bash
   pip install --no-cache-dir mujoco-py
   ```

## Testing Your Setup

### Quick Test

```bash
# Test MuJoCo availability
python -c "import mujoco_py; print('✅ MuJoCo is available')"

# Test a MuJoCo environment
python -c "import gym; env = gym.make('HalfCheetah-v2'); print('✅ HalfCheetah-v2 works')"
```

### Run Homework Without MuJoCo

All homeworks work without MuJoCo - they'll just skip those environments:

```bash
# HW2: Runs CartPole and LunarLander
cd homework/hw2
./run_all.sh

# HW3: Runs CartPole and InvertedPendulum
cd homework/hw3
./run_all.sh

# HW5: Runs Pendulum, MountainCar, and CartPole
cd homework/hw5
./run_all_hw5.sh
```

### Run Homework With MuJoCo

After installing MuJoCo, the same scripts will automatically include MuJoCo environments:

```bash
# HW2: Now also runs HalfCheetah
cd homework/hw2
./run_all.sh

# HW3: Now also runs HalfCheetah
cd homework/hw3
./run_all.sh

# HW5: Now also runs HalfCheetah and Ant
cd homework/hw5
./run_all_hw5.sh
```

## Environment Variables

If you have MuJoCo installed in a non-standard location:

```bash
# Add to ~/.bashrc or ~/.zshrc
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

## Summary

| Homework | MuJoCo Required? | Behavior Without MuJoCo |
|----------|-----------------|------------------------|
| HW2 | ❌ Optional | Trains on CartPole, LunarLander |
| HW3 | ❌ Optional | Trains on CartPole, InvertedPendulum |
| HW4 | ⚠️ Recommended | Exits with install instructions |
| HW5 | ❌ Optional | Trains on Pendulum, MountainCar, CartPole |

**Key Takeaway**: You can complete most homework assignments without MuJoCo. Install it when you want to run advanced robotics experiments!
