# CS294-112 HW1: Imitation Learning (Modular Implementation)

**Author:** Saeed Reza Zouashkiani (Student ID: 400206262)

This homework implements behavioral cloning for imitation learning with a modular, well-structured codebase.

## Project Structure

```
hw1/
├── src/
│   ├── __init__.py
│   ├── expert_data_collector.py    # Expert policy loading and data collection
│   └── behavioral_cloning.py       # Behavioral cloning training
├── run_bc.py                       # Main script for BC operations
├── run_expert.py                   # Original expert data collection script
├── load_policy.py                  # Expert policy loading utilities
├── tf_util.py                      # TensorFlow utilities
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── demo.bash                       # Demo script for testing
└── experts/                        # Expert policy files
    ├── Ant-v2.pkl
    ├── HalfCheetah-v2.pkl
    ├── Hopper-v2.pkl
    ├── Humanoid-v2.pkl
    ├── Reacher-v2.pkl
    └── Walker2d-v2.pkl
```

## Features

### Modular Components

- **ExpertDataCollector**: Handles expert policy loading and demonstration data collection
- **BehavioralCloning**: Implements the behavioral cloning algorithm with neural network training
- **Clean separation**: Data collection, training, and evaluation are separate concerns

### Functionality

- Load expert policies from pickle files
- Collect demonstration data from expert policies
- Train behavioral cloning policies using supervised learning
- Evaluate trained policies on target environments
- Support for rendering and visualization

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install MuJoCo (required for some environments):
   - Download MuJoCo 1.50 from the official website
   - Set environment variables: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mujoco150/bin`
   - Install mujoco-py: `pip install mujoco-py==1.50.1.56`

## Usage

### 1. Collect Expert Data

Use the modular script:

```bash
python run_bc.py collect --expert_policy experts/Humanoid-v2.pkl --env Humanoid-v2 --num_rollouts 20 --render
```

Or use the original script:

```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 20
```

### 2. Train Behavioral Cloning Policy

```bash
python run_bc.py train --data_file expert_data/Humanoid-v2.pkl --env Humanoid-v2 --epochs 100 --batch_size 64
```

### 3. Evaluate Trained Policy

```bash
python run_bc.py evaluate --model_file models/bc_policy.pkl --env Humanoid-v2 --episodes 10 --render
```

### 4. Run Demo

```bash
bash demo.bash
```

This will test expert policies on all available environments.

## Key Components

### ExpertDataCollector

- Loads expert policies using TensorFlow
- Collects rollouts from expert demonstrations
- Saves data in pickle format for training
- Provides statistics on expert performance

### BehavioralCloning

- Implements supervised learning on expert data
- Uses multi-layer perceptron (MLP) architecture
- Supports configurable hidden layer sizes
- Provides training history and evaluation metrics

## Environments

The implementation supports the following MuJoCo environments:

- Ant-v2
- HalfCheetah-v2
- Hopper-v2
- Humanoid-v2
- Reacher-v2
- Walker2d-v2

## Notes

- **MuJoCo Compatibility**: Recent Mac machines with NVMe disks may not be compatible with MuJoCo 1.5. Consider using Google Colab or a Linux machine for full functionality.
- **TensorFlow Version**: Uses TensorFlow 1.x for compatibility with the original expert policies.
- **Modular Design**: The new modular structure improves code maintainability while preserving all original functionality.

## References

- [CS294-112 Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [Imitation Learning Paper](https://arxiv.org/abs/1106.5327)
- [Behavioral Cloning Paper](https://www.ri.cmu.edu/pub_files/pub1/pomerleau_dean_1988_1/pomerleau_dean_1988_1.pdf)
