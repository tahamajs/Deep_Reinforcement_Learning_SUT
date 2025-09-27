# Policy Gradient Methods in Reinforcement Learning

This repository contains implementations of various policy gradient methods for reinforcement learning, organized into modular Python files for better readability and maintainability.

## Structure

- `utils.py`: Common utilities, imports, and environment setup
- `policy_gradient_visualizer.py`: Visualization tools for policy gradient concepts
- `reinforce.py`: REINFORCE algorithm implementation
- `baseline_reinforce.py`: REINFORCE with variance reduction techniques
- `actor_critic.py`: Actor-Critic and A2C implementations
- `ppo.py`: Proximal Policy Optimization (PPO) implementation
- `main.py`: Main script to run all demonstrations
- `requirements.txt`: Python dependencies

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the complete demonstration:

```bash
python main.py
```

This will execute all policy gradient method demonstrations and display performance comparisons.

## Algorithms Implemented

1. **REINFORCE**: Basic Monte Carlo policy gradient
2. **REINFORCE + Baseline**: Variance reduction using value function baseline
3. **Actor-Critic**: One-step temporal difference policy gradient
4. **A2C (Advantage Actor-Critic)**: n-step returns with advantage estimation
5. **PPO (Proximal Policy Optimization)**: Clipped surrogate objective for stable updates

## Features

- Modular code structure for easy understanding
- Comprehensive visualizations and analysis
- Performance comparisons across methods
- Training metrics and convergence analysis
- Environment: CartPole-v1 from Gymnasium

## Key Concepts Demonstrated

- Policy parameterization
- Policy gradient theorem
- Variance reduction techniques
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Trust region methods (PPO)

## Results

The implementations demonstrate the progression from basic policy gradients to state-of-the-art methods, showing improvements in:

- Sample efficiency
- Training stability
- Final performance
- Convergence speed

## Notes

- All algorithms are implemented in PyTorch
- Training uses CartPole-v1 environment
- Visualizations require matplotlib and seaborn
- GPU acceleration supported if available
