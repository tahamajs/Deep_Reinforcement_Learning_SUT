# CA9: Advanced Policy Gradient Methods

A comprehensive implementation and analysis of policy gradient methods in deep reinforcement learning, from basic REINFORCE to state-of-the-art algorithms like PPO.

## 📚 Overview

This assignment covers the progression from basic policy gradients to advanced algorithms, including:

- REINFORCE algorithm
- Variance reduction techniques (baselines)
- Actor-Critic methods
- Proximal Policy Optimization (PPO)
- Continuous control with Gaussian policies
- Hyperparameter tuning and benchmarking

## 🎯 Learning Objectives

1. **Policy Gradient Foundations**

   - Policy gradient theorem and derivation
   - REINFORCE algorithm implementation
   - Understanding variance and bias trade-offs

2. **Variance Reduction**

   - Baseline subtraction techniques
   - Advantage function estimation
   - Generalized Advantage Estimation (GAE)

3. **Actor-Critic Methods**

   - One-step actor-critic
   - Advantage Actor-Critic (A2C)
   - n-step returns

4. **Advanced Algorithms**

   - Proximal Policy Optimization (PPO)
   - Clipped surrogate objective
   - Trust region methods

5. **Continuous Control**
   - Gaussian policies
   - Action bound handling
   - Continuous action spaces

## 📁 Project Structure

```
CA9/
├── agents/                      # Agent implementations
│   ├── __init__.py
│   ├── reinforce.py            # Basic REINFORCE algorithm
│   ├── baseline_reinforce.py   # REINFORCE with variance reduction
│   ├── actor_critic.py         # Actor-Critic and A2C
│   ├── ppo.py                  # Proximal Policy Optimization
│   └── continuous_control.py   # Continuous action space agents
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── utils.py                # General utilities
│   ├── policy_gradient_visualizer.py  # Advanced visualizations
│   └── hyperparameter_tuning.py      # Hyperparameter optimization
│
├── experiments/                 # Experiment scripts
│   └── __init__.py
│
├── environments/                # Environment wrappers
│   └── __init__.py
│
├── evaluation/                  # Evaluation tools
│   └── __init__.py
│
├── models/                      # Neural network models
│   └── __init__.py
│
├── CA9.ipynb                    # Main educational notebook
├── CA9.md                       # Markdown version of notebook
├── training_examples.py         # Comprehensive training examples
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Gymnasium

### Setup

1. Clone the repository or navigate to the CA9 directory:

```bash
cd CAs/Solutions/CA9
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Main Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook CA9.ipynb
```

The notebook includes:

- Interactive policy gradient visualizations
- Step-by-step algorithm implementations
- Comparative analysis
- Performance benchmarking

### Using Individual Components

#### 1. Basic REINFORCE

```python
from agents.reinforce import REINFORCEAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = REINFORCEAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

for episode in range(500):
    reward, steps = agent.train_episode(env)
    print(f"Episode {episode}: Reward = {reward}")
```

#### 2. REINFORCE with Baseline

```python
from agents.baseline_reinforce import BaselineREINFORCEAgent

agent = BaselineREINFORCEAgent(
    state_dim,
    action_dim,
    baseline_type='value_function',  # or 'moving_average'
    lr=1e-3
)
```

#### 3. Actor-Critic

```python
from agents.actor_critic import ActorCriticAgent

agent = ActorCriticAgent(
    state_dim,
    action_dim,
    lr_actor=1e-3,
    lr_critic=1e-3,
    gamma=0.99
)
```

#### 4. PPO

```python
from agents.ppo import PPOAgent

agent = PPOAgent(
    state_dim,
    action_dim,
    lr=3e-4,
    eps_clip=0.2,
    k_epochs=4,
    buffer_size=2048
)
```

#### 5. Continuous Control

```python
from agents.continuous_control import ContinuousActorCriticAgent

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = float(env.action_space.high[0])

agent = ContinuousActorCriticAgent(
    state_dim,
    action_dim,
    action_bound=action_bound,
    lr_actor=1e-4,
    lr_critic=1e-3
)
```

### Hyperparameter Tuning

```python
from utils.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner('CartPole-v1')

# Tune learning rates
lr_results = tuner.tune_learning_rates(
    lr_range=[1e-5, 1e-4, 1e-3, 1e-2],
    num_episodes=100,
    num_seeds=3
)

# Tune PPO hyperparameters
ppo_results = tuner.tune_ppo_parameters(
    clip_ratios=[0.1, 0.2, 0.3],
    k_epochs=[3, 5, 10],
    num_episodes=150
)
```

### Comprehensive Benchmarking

```python
from utils.hyperparameter_tuning import PolicyGradientBenchmark

benchmark = PolicyGradientBenchmark()
results = benchmark.run_benchmark(num_episodes=200, num_seeds=3)
```

### Advanced Visualizations

```python
from utils.policy_gradient_visualizer import PolicyGradientVisualizer
from training_examples import (
    plot_policy_gradient_convergence_analysis,
    plot_advantage_function_analysis,
    plot_continuous_control_policy_landscapes,
    plot_hyperparameter_sensitivity_analysis,
    comprehensive_policy_gradient_comparison,
    policy_gradient_curriculum_learning,
    entropy_regularization_study,
    trust_region_policy_optimization_comparison,
    create_comprehensive_visualization_suite
)

visualizer = PolicyGradientVisualizer()

# Policy gradient intuition
visualizer.demonstrate_policy_gradient_intuition()

# Value-based vs Policy-based comparison
visualizer.compare_value_vs_policy_methods()

# Advanced visualizations
visualizer.create_advanced_visualizations()

# New comprehensive visualizations
plot_policy_gradient_convergence_analysis()
plot_advantage_function_analysis()
plot_continuous_control_policy_landscapes()
plot_hyperparameter_sensitivity_analysis()
comprehensive_policy_gradient_comparison()
policy_gradient_curriculum_learning()
entropy_regularization_study()
trust_region_policy_optimization_comparison()

# Generate all visualizations at once
create_comprehensive_visualization_suite(save_dir='visualizations/')
```

## 📊 Key Features

### 1. Modular Implementation

- Clean separation of concerns
- Easy to extend and modify
- Reusable components

### 2. Comprehensive Analysis

- Learning curve visualization
- Performance metrics tracking
- Statistical significance testing
- Hyperparameter sensitivity analysis

### 3. Educational Focus

- Detailed comments and documentation
- Step-by-step explanations
- Mathematical derivations
- Comparative insights

### 4. Production-Ready Code

- Error handling
- Gradient clipping for stability
- Proper device management (CPU/GPU)
- Configurable hyperparameters

## 🔬 Experiments

The project includes several pre-configured experiments:

1. **Variance Reduction Comparison**: Compare REINFORCE with different baseline methods
2. **Actor-Critic Variants**: Compare one-step AC, A2C with different n-steps
3. **PPO vs Alternatives**: Benchmark PPO against other policy gradient methods
4. **Continuous Control**: Test algorithms on continuous action spaces
5. **Hyperparameter Sensitivity**: Analyze impact of learning rates, clip ratios, etc.

## 📈 Results

Expected performance on CartPole-v1 (200 episodes):

- REINFORCE: ~300-400 average reward
- REINFORCE + Baseline: ~400-450 average reward
- Actor-Critic: ~450-480 average reward
- A2C: ~470-490 average reward
- PPO: ~480-500 average reward

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**

   - Ensure you're in the CA9 directory
   - Check that all `__init__.py` files are present
   - Verify Python path includes the CA9 directory

2. **CUDA/GPU Issues**

   - The code automatically detects and uses GPU if available
   - Falls back to CPU if GPU is not available
   - Set `device` manually in `utils/utils.py` if needed

3. **Training Instability**

   - Reduce learning rate
   - Increase gradient clipping threshold
   - Use a smaller batch size for PPO
   - Add entropy regularization

4. **Poor Performance**
   - Increase number of training episodes
   - Tune hyperparameters using the tuning module
   - Try different network architectures
   - Check environment rewards are properly scaled

## 📚 References

1. Sutton & Barto (2018). Reinforcement Learning: An Introduction
2. Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
3. Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning
4. Schulman et al. (2017). Proximal Policy Optimization Algorithms
5. Schulman et al. (2015). Trust Region Policy Optimization

## 🤝 Contributing

This is an educational project. Feel free to:

- Report bugs or issues
- Suggest improvements
- Add new algorithms or experiments
- Improve documentation

## 📄 License

This project is part of a Deep Reinforcement Learning course assignment.

## 👥 Authors

- Course: Deep Reinforcement Learning
- Session: 9
- Topic: Advanced Policy Gradient Methods

## 🙏 Acknowledgments

- OpenAI Gymnasium for the environments
- PyTorch team for the deep learning framework
- The reinforcement learning community for research and insights

---

**Note**: This implementation is for educational purposes. For production use, consider using established libraries like Stable-Baselines3 or RLlib.
