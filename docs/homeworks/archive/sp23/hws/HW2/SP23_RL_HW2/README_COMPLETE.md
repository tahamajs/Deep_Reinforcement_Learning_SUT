# PPO vs DDPG Implementation - Complete Notebook

## 📚 Overview

This notebook provides a **complete implementation** of two state-of-the-art deep reinforcement learning algorithms:
- **PPO (Proximal Policy Optimization)**
- **DDPG (Deep Deterministic Policy Gradient)**

Both algorithms are trained and compared on the **Pendulum-v1** environment from OpenAI Gym.

## ✨ What's Implemented

### Core Components

#### 1. **Replay Buffer** (`ReplayBuffer`)
- Efficient numpy-based circular buffer
- Stores transitions: (state, action, reward, next_state, done)
- Random mini-batch sampling for breaking temporal correlations
- O(1) indexing performance

#### 2. **OU Noise** (`OUNoise`)
- Ornstein-Uhlenbeck process for temporally correlated exploration
- Used by DDPG for smooth, physics-friendly exploration
- Configurable parameters: μ, θ, σ

#### 3. **PPO Networks**
- **PPOActor**: Stochastic policy network
  - Outputs mean (μ) and log-std (log σ) for Gaussian distribution
  - Tanh activation for bounded actions
  - Samples actions from 𝒩(μ, σ²)
  
- **Critic**: State-value function network
  - Estimates V(s)
  - Used for computing advantages

#### 4. **DDPG Networks**
- **DDPGActor**: Deterministic policy network
  - Outputs deterministic action μ(s)
  - Tanh activation for bounded actions
  
- **DDPGCritic**: Action-value function network
  - Estimates Q(s, a)
  - Takes concatenated state-action input

#### 5. **PPO Agent** (`PPOAgent`)
- **Generalized Advantage Estimation (GAE)**
  - Computes advantages with bias-variance trade-off
  - Configurable τ parameter
  
- **Clipped Surrogate Objective**
  - Prevents destructive policy updates
  - Clips probability ratio to [1-ε, 1+ε]
  
- **Entropy Regularization**
  - Encourages exploration
  - Weighted entropy bonus in loss
  
- **Multiple Epochs**
  - Reuses collected data multiple times
  - Improves sample efficiency while maintaining stability

#### 6. **DDPG Agent** (`DDPGAgent`)
- **Deterministic Policy Gradient**
  - Maximizes Q(s, μ(s))
  - Chain rule for gradient computation
  
- **Target Networks**
  - Separate target networks for stability
  - Soft updates: θ' ← τθ + (1-τ)θ'
  
- **Experience Replay**
  - Stores all past experiences
  - Random mini-batch sampling
  
- **Exploration Strategy**
  - Initial random exploration
  - OU noise added during training

#### 7. **Action Normalizer** (`ActionNormalizer`)
- Gym wrapper for action space normalization
- Maps network output [-1, 1] to environment range [low, high]
- Bidirectional transformation (forward and reverse)

## 🎯 Key Features

### Educational Value
- ✅ **Comprehensive comments** explaining each component
- ✅ **Detailed markdown cells** with mathematical formulations
- ✅ **Side-by-side comparison** of PPO and DDPG
- ✅ **Implementation best practices** for deep RL
- ✅ **Visualization functions** for training progress

### Code Quality
- ✅ **Type hints** for better code readability
- ✅ **Modular design** with clear separation of concerns
- ✅ **Proper initialization** of network weights
- ✅ **Device handling** (CPU/GPU support)
- ✅ **Seed setting** for reproducibility

### Algorithm Implementations
- ✅ **GAE** for advantage estimation (PPO)
- ✅ **Clipped objective** for stable policy updates (PPO)
- ✅ **Entropy regularization** for exploration (PPO)
- ✅ **Target networks** for stability (DDPG)
- ✅ **Soft updates** for smooth learning (DDPG)
- ✅ **Replay buffer** for sample efficiency (DDPG)

## 📊 PPO vs DDPG Comparison

| Aspect | PPO | DDPG |
|--------|-----|------|
| **Type** | On-policy | Off-policy |
| **Policy** | Stochastic | Deterministic |
| **Exploration** | Intrinsic (entropy) | Extrinsic (noise) |
| **Stability** | High (clipping) | Medium (needs tuning) |
| **Sample Efficiency** | Lower | Higher |
| **Value Function** | V(s) | Q(s,a) |
| **Memory** | Short-term rollouts | Long-term replay buffer |
| **Updates** | Multiple epochs | Every step |

### When to Use?

**Use PPO when:**
- ✅ Stability is critical
- ✅ You want robust performance across diverse tasks
- ✅ Hyperparameter tuning time is limited
- ✅ On-policy learning is acceptable

**Use DDPG when:**
- ✅ Sample efficiency is important
- ✅ Deterministic policies are preferred
- ✅ You have time for hyperparameter tuning
- ✅ Off-policy learning is beneficial

## 🚀 Getting Started

### Prerequisites

```bash
pip install gym
pip install torch
pip install numpy
pip install matplotlib
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

### Running the Notebook

1. **Open the notebook:**
   ```bash
   jupyter notebook RL_HW2_PPO_vs_DDPG_COMPLETE.ipynb
   ```

2. **Run all cells sequentially** (Cell → Run All)

3. **Watch the training progress:**
   - Score plots show learning curves
   - Actor loss shows policy optimization
   - Critic loss shows value function learning

4. **Test the trained agents:**
   - Both agents will be tested after training
   - Rendered videos show the learned behavior

### Expected Results

#### Pendulum-v1
- **Untrained agent**: ~-1200 to -1500 reward
- **Trained PPO**: ~-200 to -150 reward
- **Trained DDPG**: ~-150 to -100 reward

Both agents should learn to:
1. Swing the pendulum up
2. Balance it at the top
3. Minimize energy consumption

## 🎓 Learning Objectives

After completing this notebook, you will understand:

1. **Fundamental RL Concepts:**
   - Actor-Critic architecture
   - Policy gradients
   - Value function estimation
   - Advantage functions

2. **Advanced RL Techniques:**
   - Generalized Advantage Estimation (GAE)
   - Clipped surrogate objective
   - Target networks and soft updates
   - Experience replay

3. **Implementation Details:**
   - Network architecture design
   - Hyperparameter selection
   - Training loop organization
   - Debugging and visualization

4. **Algorithm Comparison:**
   - On-policy vs off-policy
   - Stochastic vs deterministic policies
   - Stability vs sample efficiency
   - When to use which algorithm

## 📖 Mathematical Background

### PPO Objective
The clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

### DDPG Objective
Actor loss (deterministic policy gradient):

$$\nabla_\theta J = \mathbb{E}_{s \sim \rho} \left[ \nabla_a Q(s,a|theta^Q)|_{a=\mu(s)} \nabla_\theta \mu(s|\theta^\mu) \right]$$

Critic loss (Bellman equation):

$$L = \mathbb{E} \left[ (Q(s,a) - (r + \gamma Q'(s', \mu'(s'))))^2 \right]$$

### GAE (Generalized Advantage Estimation)
Used in PPO for computing advantages:

$$\hat{A}_t^{GAE(\gamma, \tau)} = \sum_{l=0}^{\infty} (\gamma \tau)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

## 🔧 Hyperparameters

### PPO Hyperparameters
```python
gamma = 0.9              # Discount factor
tau = 0.8                # GAE parameter
batch_size = 64          # Mini-batch size
epsilon = 0.2            # Clipping range
epoch = 64               # Number of update epochs
rollout_len = 2048       # Rollout length
entropy_weight = 0.005   # Entropy coefficient
```

### DDPG Hyperparameters
```python
gamma = 0.99             # Discount factor
tau = 5e-3               # Soft update coefficient
batch_size = 128         # Mini-batch size
memory_size = 20000      # Replay buffer size
ou_noise_theta = 1.0     # OU noise mean reversion
ou_noise_sigma = 0.1     # OU noise volatility
initial_random_steps = 10000  # Warmup steps
```

## 📝 Code Structure

```
RL_HW2_PPO_vs_DDPG_COMPLETE.ipynb
├── Configuration & Setup
├── Module Imports
├── Random Seed Setting
├── Replay Buffer Implementation
├── OU Noise Implementation
├── Network Architectures
│   ├── PPOActor
│   ├── Critic
│   ├── DDPGActor
│   └── DDPGCritic
├── Agent Implementations
│   ├── ppo_iter (mini-batch generator)
│   ├── PPOAgent
│   └── DDPGAgent
├── Environment Setup
│   └── ActionNormalizer
├── Training & Testing
│   ├── Initialize Agents
│   ├── Train PPO
│   ├── Train DDPG
│   ├── Test Both Agents
│   └── Render Results
└── Comparison & Analysis
```

## 🎨 Visualization

The notebook includes:
- **Training curves**: Score, actor loss, critic loss
- **Real-time plotting**: Updated every N frames
- **Test videos**: Rendered gameplay from trained agents
- **Comparison plots**: Side-by-side performance

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size`
   - Reduce `memory_size`
   - Use CPU: Remove `.to(device)` calls

2. **Agent not learning**
   - Check learning rates (try 1e-4 to 1e-3)
   - Increase `num_frames`
   - Adjust exploration (PPO: `entropy_weight`, DDPG: noise parameters)

3. **Training unstable**
   - Reduce learning rates
   - Increase `tau` (PPO GAE parameter)
   - Decrease `tau` (DDPG soft update)

4. **Import errors**
   - Install missing packages: `pip install gym torch matplotlib`
   - For older gym versions, use `gym.make("Pendulum-v0")`

## 📚 References

### Papers
1. **PPO**: [Schulman et al., 2017 - Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. **DDPG**: [Lillicrap et al., 2015 - Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
3. **GAE**: [Schulman et al., 2016 - High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
4. **TRPO**: [Schulman et al., 2015 - Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

### Resources
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gym Documentation](https://www.gymlibrary.dev/)
- [PyTorch Deep RL Tutorials](https://pytorch.org/tutorials/)

## 📄 License

This educational material is provided for learning purposes. Please refer to the course materials for specific licensing information.

## 🤝 Contributing

This is a complete implementation for educational purposes. If you find any issues or have suggestions for improvements:
1. Test your changes thoroughly
2. Ensure code quality and readability
3. Add appropriate comments and documentation

## ✍️ Author

Completed as part of Deep Reinforcement Learning course assignments.

## 🌟 Acknowledgments

- OpenAI for Gym and Spinning Up resources
- Original paper authors (Schulman, Lillicrap, et al.)
- Deep RL community for best practices and implementations

---

**Happy Learning! 🚀**

For questions or issues, please refer to the course materials or consult the teaching assistants.

