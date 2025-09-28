# CS294-112 HW 3: Q-Learning (DQN) and Actor-Critic

**Author:** Saeed Reza Zouashkiani - Student ID: 400206262

This homework implements Deep Q-Learning (DQN) with experience replay and target networks, as well as Actor-Critic methods for reinforcement learning.

## Project Structure

```
hw3/
├── src/
│   ├── dqn.py              # DQN agent with experience replay and target networks
│   └── actor_critic.py     # Actor-Critic agent with policy and value networks
├── run_dqn_atari.py        # DQN training script for Atari environments
├── run_dqn_lander.py       # DQN training script for LunarLander
├── run_ac.py               # Actor-Critic training script
├── dqn_utils.py            # DQN utilities (replay buffer, schedules)
├── atari_wrappers.py       # Atari environment wrappers
├── logz.py                 # Logging utilities
├── lunar_lander.py         # Modified lunar lander environment
├── plot.py                 # Plotting utilities
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── results/               # Training results and logs
```

## Dependencies

* Python **3.7+**
* NumPy **1.21.0+**
* TensorFlow **1.10.5+** (or TensorFlow 2.x with compatibility)
* MuJoCo **2.1.0+** and mujoco-py **2.1.0+**
* OpenAI Gym **0.21.0+**
* seaborn
* Box2D **2.3.10+**
* OpenCV
* ffmpeg

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

### Deep Q-Learning (DQN)

#### Atari Games
```bash
python run_dqn_atari.py PongNoFrameskip-v4 --num_timesteps 2000000 --double_q
```

#### LunarLander
```bash
python run_dqn_lander.py LunarLander-v2 --num_timesteps 50000
```

### Actor-Critic
```bash
python run_ac.py CartPole-v0 --n_iter 100 --batch_size 1000 --learning_rate 5e-3
```

## Key Arguments

### DQN Arguments
* `env_name`: Gym environment name
* `--num_timesteps`: Number of timesteps to train
* `--seed`: Random seed
* `--double_q`: Use double Q-learning (Atari only)

### Actor-Critic Arguments
* `env_name`: Gym environment name
* `--n_iter`: Number of training iterations
* `--batch_size`: Minimum timesteps per batch
* `--learning_rate`: Learning rate
* `--discount`: Discount factor gamma
* `--num_target_updates`: Number of critic target updates
* `--num_grad_steps_per_target_update`: Gradient steps per target update
* `--normalize_advantages`: Normalize advantages
* `--n_layers`: Number of hidden layers
* `--size`: Size of hidden layers

## Implementation Details

### Modular Components

1. **dqn.py**: DQN Agent
   - Experience replay buffer
   - Target Q-networks for stable learning
   - Double Q-learning option
   - Huber loss for robust Q-learning
   - Support for both image and vector observations

2. **actor_critic.py**: Actor-Critic Agent
   - Separate policy and value networks
   - Advantage estimation with critic baseline
   - Support for discrete and continuous action spaces
   - Bootstrapped target updates for critic

### Key Features

* **Experience Replay**: Stores and samples past transitions for stable learning
* **Target Networks**: Separate networks for stable Q-value targets
* **Double Q-Learning**: Reduces overestimation bias in Q-values
* **Huber Loss**: Robust loss function less sensitive to outliers
* **Advantage Normalization**: Stabilizes policy gradient updates

## Results

Training results and logs are saved in the `results/` directory. Use `plot.py` to visualize learning curves:

```bash
python plot.py results/experiment_name/
```

## Atari Environment Setup

For Atari games, the environment is automatically wrapped with:
- Frame skipping (4 frames)
- Frame stacking (4 consecutive frames)
- Grayscale conversion
- Frame resizing to 84x84
- Reward clipping

## References

* [CS294-112 Homework 3 PDF](cs285_hw3.pdf)
* Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
* Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. ICML.
* Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv.
