# Complete Guide to Running Behavioral Cloning

## Quick Start

### Option 1: Run a Single Environment (Recommended for Testing)

```bash
cd /Users/tahamajs/Documents/uni/DRL/Other_Assisments/homework/hw1

# Run the complete pipeline for Hopper-v2
python run_full_pipeline.py --env Hopper-v2 --num_rollouts 20 --epochs 100
```

### Option 2: Run All Environments

```bash
cd /Users/tahamajs/Documents/uni/DRL/Other_Assisments/homework/hw1

# Make the script executable
chmod +x run_all_environments.sh

# Run all environments
./run_all_environments.sh
```

### Option 3: Step-by-Step (Using Modular Components)

```bash
# Step 1: Collect expert data
python run_bc.py collect --env Hopper-v2 --expert_policy experts/Hopper-v2.pkl --num_rollouts 20

# Step 2: Train BC policy
python run_bc.py train --env Hopper-v2 --data_file expert_data/Hopper-v2.pkl --epochs 100 --batch_size 64

# Step 3: Evaluate BC policy
python run_bc.py evaluate --env Hopper-v2 --model_file models/bc_policy --episodes 10 --render
```

## Available Environments

- **Hopper-v2** - 2D hopping robot (fastest, good for testing)
- **Ant-v2** - 3D quadruped robot
- **HalfCheetah-v2** - 2D running robot
- **Walker2d-v2** - 2D walking robot
- **Reacher-v2** - 2D arm reaching task
- **Humanoid-v2** - 3D humanoid robot (slowest, most complex)

## Parameters

### Data Collection
- `--num_rollouts`: Number of expert demonstrations (default: 20)
  - More rollouts = more data = potentially better performance
  - Typical range: 10-50

### Training
- `--epochs`: Number of training epochs (default: 100)
  - More epochs = more training time
  - Typical range: 50-200
- `--batch_size`: Batch size for training (default: 64)
  - Larger batch = more stable gradients
  - Typical range: 32-128
- `--learning_rate`: Learning rate (default: 1e-3)
  - Typical range: 1e-4 to 1e-2

### Evaluation
- `--episodes`: Number of evaluation episodes (default: 10)
- `--render`: Show visualization (slower but visible)

## Expected Results

Typical performance ratios (BC policy / Expert policy):
- **Hopper-v2**: 70-90%
- **Reacher-v2**: 80-95%
- **HalfCheetah-v2**: 60-80%
- **Walker2d-v2**: 60-80%
- **Ant-v2**: 50-70%
- **Humanoid-v2**: 40-60%

## Troubleshooting

### TensorFlow Compatibility Issues

If you encounter TensorFlow errors, this code uses TensorFlow 1.x. You may need to:

```bash
# Create a separate environment
conda create -n bc_env python=3.6
conda activate bc_env

# Install dependencies
pip install tensorflow==1.15.0
pip install gym==0.10.5
pip install mujoco-py==1.50.1.56
pip install numpy matplotlib seaborn
```

### MuJoCo Issues

MuJoCo 1.50 is required. If you have issues:

1. Download MuJoCo 1.50 from the official website
2. Place it in `~/.mujoco/mujoco150`
3. Get a license key and place it in `~/.mujoco/mjkey.txt`
4. Set environment variable:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco150/bin
   ```

### Mac Compatibility

On newer Macs (especially M1/M2), MuJoCo 1.50 may not work. Consider:
- Using Google Colab
- Using a Docker container with Linux
- Upgrading to newer versions of gym and mujoco-py (requires code changes)

## Output Files

After running, you'll find:

```
hw1/
├── expert_data/          # Collected expert demonstrations
│   ├── Hopper-v2.pkl
│   ├── Ant-v2.pkl
│   └── ...
├── models/               # Trained BC policies
│   ├── bc_policy_Hopper-v2.data-00000-of-00001
│   ├── bc_policy_Hopper-v2.index
│   ├── bc_policy_Hopper-v2.meta
│   ├── training_history_Hopper-v2.pkl
│   └── ...
```

## Testing Without Full Run

To quickly test the expert policies without training:

```bash
# Test a single environment
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts 1

# Test all environments
bash demo.bash
```

## Performance Tips

1. **Start small**: Test with Hopper-v2 or Reacher-v2 first (fastest)
2. **Use fewer rollouts initially**: Try 10 rollouts to test the pipeline
3. **Monitor training**: Watch the loss decrease over epochs
4. **Compare results**: BC should get 50-90% of expert performance

## Common Commands Summary

```bash
# Quick test (Hopper, no rendering)
python run_full_pipeline.py --env Hopper-v2 --num_rollouts 10 --epochs 50

# Full run (Hopper, with rendering during evaluation)
python run_full_pipeline.py --env Hopper-v2 --num_rollouts 20 --epochs 100 --render_eval

# Process all environments (no rendering)
./run_all_environments.sh

# Test expert policy only
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts 5
```

## Notes

- Training time varies by environment: Hopper (~5 min), Humanoid (~30 min)
- Rendering significantly slows down execution
- Results may vary between runs due to randomness
- Performance depends on number of expert demonstrations and training epochs

## Contact

Author: Saeed Reza Zouashkiani
Student ID: 400206262
Course: CS294-112 Deep Reinforcement Learning
