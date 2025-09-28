# Assignment 2: Value Iteration, Policy Iteration, and Deep Q-Networks

Author: Taha Majlesi - 810101504, University of Tehran

This assignment implements various reinforcement learning algorithms: Value Iteration (VI), Policy Iteration (PI), and Deep Q-Networks (DQN).

## Structure

- `src/utils.py`: Utility functions for policy printing and value function to policy conversion
- `src/environment.py`: Environment wrapper for FrozenLake
- `src/policy_evaluation.py`: Policy evaluation algorithms (synchronous, async ordered, async random permutation)
- `src/policy_improvement.py`: Policy improvement algorithm
- `src/policy_iteration.py`: Policy iteration algorithms (synchronous, async ordered, async random permutation)
- `src/value_iteration.py`: Value iteration algorithms (synchronous, async ordered, async random permutation)
- `src/visualization.py`: Visualization functions for policies and value functions
- `src/q_network.py`: Q-network architecture for DQN
- `src/replay_memory.py`: Experience replay memory for DQN
- `src/dqn_agent.py`: DQN agent implementation
- `run_vi_pi.py`: Script to run VI and PI algorithms
- `run_dqn.py`: Script to run DQN algorithm
- `requirements.txt`: Python dependencies

## Algorithms

### Value Iteration (VI)
- **Synchronous VI**: Updates all states simultaneously
- **Async Ordered VI**: Updates states in 1-N order
- **Async Random Permutation VI**: Updates states in random order

### Policy Iteration (PI)
- **Synchronous PI**: Alternates between policy evaluation and improvement
- **Async Ordered PI**: Uses async policy evaluation with ordered updates
- **Async Random Permutation PI**: Uses async policy evaluation with random permutation updates

### Deep Q-Networks (DQN)
- Experience replay memory
- Target network for stable training
- Epsilon-greedy exploration
- Support for Double DQN

## Environments

- **FrozenLake**: 4x4 and 8x8 grid worlds for VI/PI algorithms
- **CartPole-v0**: Classic control task for DQN
- **MountainCar-v0**: Continuous control task for DQN

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Value Iteration and Policy Iteration
```bash
python run_vi_pi.py
```

### Deep Q-Networks
```bash
# Train DQN on CartPole
python run_dqn.py --env CartPole-v0

# Train Double DQN on MountainCar
python run_dqn.py --env MountainCar-v0 --double_dqn 1

# Render trained agent
python run_dqn.py --env CartPole-v0 --render --model path/to/model.h5
```

## Dependencies

- gymnasium: Modern RL environments
- numpy: Numerical computations
- scipy: Scientific computing
- matplotlib: Plotting and visualization
- keras: Neural network framework
- tensorflow: Deep learning backend
- tensorboardX: Logging and visualization
- seaborn: Statistical visualization

## Results

The algorithms are evaluated on convergence speed and final performance. VI and PI find optimal policies for small MDPs, while DQN learns effective policies for continuous control tasks.