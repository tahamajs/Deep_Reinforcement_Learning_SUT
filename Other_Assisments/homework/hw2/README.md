# CS294-112 HW 2: Policy Gradient

**Author:** Saeed Reza Zouashkiani - Student ID: 400206262

This homework implements policy gradient methods with optional neural network baselines for reinforcement learning.

## Project Structure

```
hw2/
├── src/
│   ├── networks.py          # Neural network architectures (PolicyNetwork, ValueNetwork)
│   ├── policy_gradient.py   # Main policy gradient agent implementation
│   └── utils.py             # Trajectory processing utilities
├── run_pg.py                # Main training script
├── logz.py                  # Logging utilities
├── lunar_lander.py          # Modified lunar lander environment
├── plot.py                  # Plotting utilities
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── results/                 # Training results and logs
```

## Dependencies

- Python **3.7+**
- NumPy **1.21.0+**
- TensorFlow **2.8.0+**
- MuJoCo **2.1.0+** and mujoco-py **2.1.0+**
- OpenAI Gym **0.21.0+**
- seaborn
- Box2D **2.3.10+**

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Replace the default lunar lander environment:
   ```bash
   cp lunar_lander.py /path/to/gym/envs/box2d/lunar_lander.py
   ```

## Usage

### Training a Policy Gradient Agent

Run the main training script:

```bash
python run_pg.py CartPole-v0 --n_iter 100 --batch_size 1000 --learning_rate 5e-3
```

### Key Arguments

- `env_name`: Gym environment name (e.g., CartPole-v0, LunarLander-v2)
- `--n_iter`: Number of training iterations (default: 100)
- `--batch_size`: Minimum timesteps per batch (default: 1000)
- `--learning_rate`: Learning rate for optimization (default: 5e-3)
- `--discount`: Discount factor gamma (default: 1.0)
- `--reward_to_go`: Use reward-to-go for advantage estimation
- `--nn_baseline`: Use neural network baseline for variance reduction
- `--dont_normalize_advantages`: Disable advantage normalization
- `--n_layers`: Number of hidden layers in networks (default: 2)
- `--size`: Size of hidden layers (default: 64)
- `--render`: Render episodes during training

### Examples

Train on CartPole with reward-to-go and neural baseline:

```bash
python run_pg.py CartPole-v0 --reward_to_go --nn_baseline --n_iter 50
```

Train on LunarLander with custom learning rate:

```bash
python run_pg.py LunarLander-v2 --learning_rate 1e-3 --batch_size 2000 --n_iter 200
```

## Implementation Details

### Modular Components

1. **networks.py**: Contains neural network architectures

   - `build_mlp()`: Multi-layer perceptron builder
   - `PolicyNetwork`: Policy network for action selection
   - `ValueNetwork`: Value network for baseline estimation

2. **policy_gradient.py**: Main agent implementation

   - `PolicyGradientAgent`: Core agent class with trajectory sampling and training
   - Supports both discrete and continuous action spaces
   - Implements reward-to-go and trajectory-based returns
   - Optional neural network baseline for variance reduction

3. **utils.py**: Trajectory processing utilities
   - Path length calculation
   - Trajectory flattening for batch processing
   - Advantage computation with various options
   - Q-value estimation

### Key Features

- **Reward-to-Go**: More efficient credit assignment by only considering future rewards
- **Neural Network Baseline**: Reduces variance using learned value function
- **Advantage Normalization**: Stabilizes training by normalizing advantages
- **Flexible Network Architecture**: Configurable hidden layers and sizes
- **Multi-Environment Support**: Works with both discrete and continuous action spaces

## Results

Training results and logs are saved in the `results/` directory. Use `plot.py` to visualize learning curves:

```bash
python plot.py results/experiment_name/
```

## References

- [CS294-112 Homework 2 PDF](cs285_hw2.pdf)
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. ICML.
