# HW4: Actor-Critic Methods

## Quick Run Sheet
1) **Env**: Python 3.8–3.11 (fresh venv).  
2) **Install deps**:  
```bash
pip install -r code/requirements.txt        # main notebook env (if provided)
pip install -r base_code/requirements.txt   # starter env (optional)
```  
3) **Work notebooks**: see `code/` for student versions; `base_code/` for minimal scaffolding.  
4) **Prompt**: question PDF under `reports/` (or `HW4_Complete_Solution.md` for expectations).  
5) **Reference**: solutions in `answers/` and `HW4_Complete_Solution.md`; template in `Homework-4-Template*`.  
6) **Deliverables**: completed notebook(s) + PDF with learning curves and hyperparameters.

## Folder Legend
- `code/` – main notebooks/scripts with TODOs.  
- `base_code/` – starter version.  
- `answers/` – filled solutions (view after attempt).  
- `reports/` – questions/rubric.  
- `HW4_Complete_Solution.md` – narrative solution walk-through.  
- `Homework-4-Template*` – release template (folder + zip).

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Actor-Critic](https://img.shields.io/badge/Methods-Actor--Critic-red.svg)](https://www.deepmind.com/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment explores **Actor-Critic methods**, which combine the best of both policy-based and value-based approaches. These methods use two neural networks: an **actor** that learns the policy and a **critic** that learns the value function to guide the actor's learning. We'll implement state-of-the-art algorithms including PPO, SAC, and DDPG.

## 🎯 Learning Objectives

By completing this assignment, you will:

1. **Understand Actor-Critic Architecture**: Learn how actor and critic networks collaborate
2. **Master PPO**: Implement Proximal Policy Optimization with clipped objectives
3. **Continuous Control with DDPG**: Learn deterministic policy gradients
4. **Soft Actor-Critic (SAC)**: Understand maximum entropy RL and automatic temperature tuning
5. **On-Policy vs Off-Policy**: Compare PPO (on-policy) with SAC/DDPG (off-policy)
6. **Advanced Techniques**: Trust regions, target networks, and entropy regularization

## 📂 Directory Structure

```
HW4_Actor_Critic/
├── code/
│   ├── HW4_P1_PPO_Continuous.ipynb           # Proximal Policy Optimization
│   └── HW4_P2_SAC_DDPG_Continuous.ipynb      # SAC and DDPG comparison
├── answers/
│   ├── HW4_P1_PPO_Continuous_Solution.ipynb
│   ├── HW4_P2_SAC_DDPG_Continuous_Solution.ipynb
│   └── HW4_Solution.pdf
├── reports/
│   └── HW4_Questions.pdf
└── README.md
```

## 📚 Theoretical Background

### 1. Actor-Critic Framework

**Core Idea:** Combine policy gradient (actor) with value function approximation (critic).

```
Actor:  πθ(a|s) - Selects actions
Critic: Vφ(s) or Qφ(s,a) - Evaluates actions
```

**Advantages:**
- Lower variance than pure policy gradients (critic reduces variance)
- More sample efficient than REINFORCE
- Can be on-policy (A2C, PPO) or off-policy (DDPG, SAC)
- Natural baseline through value function

**Update Rules:**
```
# Critic update (TD learning)
δt = rt + γV(st+1) - V(st)  # TD error
φ ← φ + αc δt ∇φ V(st)

# Actor update (policy gradient with advantage)
θ ← θ + αa δt ∇θ log π(at|st)
```

### 2. Proximal Policy Optimization (PPO)

**Motivation:** Policy gradients can take large, destructive updates. PPO constrains updates to stay close to the current policy.

**Two Variants:**

#### a) PPO-Penalty
```
L(θ) = 𝔼[L^CPI(θ) - β KL[πθ_old, πθ]]
```

#### b) PPO-Clip (Most Common)
```
L^CLIP(θ) = 𝔼[min(rt(θ)Ât, clip(rt(θ), 1-ε, 1+ε)Ât)]

where rt(θ) = πθ(at|st) / πθ_old(at|st)  # Importance sampling ratio
```

**Key Features:**
- **Clipping**: Limits how much policy can change
- **Multiple Epochs**: Reuse same batch for multiple updates
- **Generalized Advantage Estimation (GAE)**: Balances bias-variance
  ```
  Ât = ∑k=0^∞ (γλ)^k δt+k
  where δt = rt + γV(st+1) - V(st)
  ```

**Pseudocode:**
```python
for iteration in range(N):
    # Collect trajectories using πθ_old
    trajectories = collect_trajectories(πθ_old)
    
    # Compute advantages using GAE
    advantages = compute_gae(trajectories)
    
    # Multiple epochs on same data
    for epoch in range(K):
        for batch in minibatches(trajectories):
            # Compute ratio
            ratio = π(a|s) / π_old(a|s)
            
            # PPO-Clip objective
            L_clip = min(ratio * advantage, 
                        clip(ratio, 1-ε, 1+ε) * advantage)
            
            # Optimize actor and critic
            loss = -L_clip + c1*L_value - c2*entropy
            optimize(loss)
    
    # Update old policy
    θ_old ← θ
```

**Why PPO Works:**
- Prevents catastrophic policy updates
- Allows multiple gradient steps per data collection
- More sample efficient than vanilla policy gradient
- Simpler than TRPO, similar performance

### 3. Deep Deterministic Policy Gradient (DDPG)

**Key Idea:** Extend DQN to continuous action spaces using deterministic policies.

**Architecture:**
```
Actor:  μθ(s) → a  (deterministic policy)
Critic: Qφ(s,a) → ℝ (action-value function)
```

**Deterministic Policy Gradient Theorem:**
```
∇θJ ≈ 𝔼[∇a Q(s,a)|a=μθ(s) ∇θ μθ(s)]
```

**Key Components:**

#### a) Experience Replay
Store transitions (s, a, r, s') and sample randomly for training.

#### b) Target Networks
```
θ' ← τθ + (1-τ)θ'  # Soft update
φ' ← τφ + (1-τ)φ'
```

#### c) Exploration Noise
```
a = μθ(s) + ε, where ε ~ N(0, σ)  # Gaussian noise
```

**DDPG Algorithm:**
```python
# Initialize networks
actor = μθ(s)
critic = Qφ(s,a)
target_actor = μθ'(s)
target_critic = Qφ'(s,a)

for episode in range(N):
    for t in range(T):
        # Select action with exploration
        a = μθ(s) + ε
        
        # Execute and store
        s', r = env.step(a)
        buffer.store(s, a, r, s')
        
        # Sample batch
        batch = buffer.sample()
        
        # Critic update
        y = r + γ Qφ'(s', μθ'(s'))
        L_critic = MSE(Qφ(s,a), y)
        
        # Actor update
        L_actor = -𝔼[Qφ(s, μθ(s))]
        
        # Soft target update
        θ' ← τθ + (1-τ)θ'
        φ' ← τφ + (1-τ)φ'
```

**Challenges:**
- Sensitive to hyperparameters
- Can overestimate Q-values
- Exploration in continuous spaces difficult

### 4. Soft Actor-Critic (SAC)

**Motivation:** Maximum entropy RL - maximize expected return AND policy entropy.

**Objective:**
```
J(θ) = ∑t 𝔼[rt + α H(πθ(·|st))]

where H(π) = -𝔼[log π(a|s)]  # Entropy
      α = temperature parameter
```

**Why Maximum Entropy?**
- Encourages exploration automatically
- More robust to reward scale
- Prevents premature convergence
- Learns multi-modal policies

**Key Features:**

#### a) Stochastic Policy
```python
class SAC_Actor(nn.Module):
    def forward(self, state):
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample with gradient
        action = torch.tanh(x_t)  # Squash to [-1, 1]
        
        # Log probability with change of variables
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        
        return action, log_prob
```

#### b) Twin Q-Networks
Minimize overestimation using two Q-functions:
```
Q_target = r + γ(min(Q1', Q2') - α log π)
```

#### c) Automatic Temperature Tuning
Learn α automatically to match target entropy:
```
L_α = -𝔼[α(log π(a|s) + H_target)]
```

**SAC Algorithm:**
```python
# Initialize
actor = πθ(a|s)
Q1, Q2 = Qφ1(s,a), Qφ2(s,a)
Q1_target, Q2_target = Qφ1'(s,a), Qφ2'(s,a)
log_α = learnable parameter

for step in range(N):
    # Sample action
    a, log_prob = actor.sample(s)
    
    # Execute
    s', r = env.step(a)
    buffer.store(s, a, r, s', log_prob)
    
    # Sample batch
    batch = buffer.sample()
    
    # Critic update
    with torch.no_grad():
        a_next, log_prob_next = actor.sample(s')
        Q_next = min(Q1_target(s', a_next), 
                     Q2_target(s', a_next))
        y = r + γ(Q_next - α * log_prob_next)
    
    L_Q = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
    
    # Actor update
    a_new, log_prob_new = actor.sample(s)
    Q_new = min(Q1(s, a_new), Q2(s, a_new))
    L_actor = 𝔼[α * log_prob_new - Q_new]
    
    # Temperature update
    L_α = -𝔼[log_α * (log_prob + H_target)]
    
    # Soft target update
    update_targets(τ)
```

**Advantages of SAC:**
- State-of-the-art continuous control
- Robust to hyperparameters
- Automatic exploration via entropy
- Sample efficient (off-policy)

### 5. Comparison Table

| Algorithm | Policy Type | On/Off-Policy | Key Feature | Best For |
|-----------|-------------|---------------|-------------|----------|
| **PPO** | Stochastic | On-Policy | Trust region | Robotics, General |
| **DDPG** | Deterministic | Off-Policy | DPG + DQN | Continuous control |
| **SAC** | Stochastic | Off-Policy | Max entropy | Continuous, Exploration |

**When to Use:**
- **PPO**: Default choice, stable, good performance
- **DDPG**: Need deterministic policy, simpler than SAC
- **SAC**: Need best sample efficiency, complex exploration

## 💻 Implementation Details

### Part 1: PPO for Continuous Control

**Environment:** Continuous control tasks (Pendulum, HalfCheetah, Ant)

**Network Architecture:**
```python
class PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

class PPO_Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)
```

**Hyperparameters:**
- Clip ratio ε: 0.2
- GAE λ: 0.95
- Epochs per update: 10
- Mini-batch size: 64
- Learning rate: 3e-4
- Value coefficient: 0.5
- Entropy coefficient: 0.01

### Part 2: SAC and DDPG Comparison

**Tasks:**
1. Implement DDPG with Ornstein-Uhlenbeck noise
2. Implement SAC with automatic temperature
3. Compare sample efficiency and final performance
4. Analyze exploration strategies

**DDPG Exploration Noise:**
```python
class OUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = copy.copy(self.mu)
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state
```

## 📊 Evaluation Metrics

1. **Episode Return**: Average return over evaluation episodes
2. **Sample Efficiency**: Return vs environment steps
3. **Training Stability**: Variance across random seeds
4. **Exploration**: Policy entropy over time (for stochastic policies)
5. **Q-Value Accuracy**: Difference between estimated and actual returns
6. **Wall-Clock Time**: Real training time

## 🔧 Requirements

```python
# Core Libraries
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# RL Environments
gymnasium>=0.28.0
pybullet>=3.2.0  # For robotics environments
mujoco>=2.3.0    # For MuJoCo environments

# Deep Learning
torch>=2.0.0

# Utilities
pandas>=1.3.0
tqdm>=4.62.0
tensorboard>=2.9.0
```

## 🚀 Getting Started

```bash
cd HW4_Actor_Critic
pip install -r requirements.txt

# Run PPO
jupyter notebook code/HW4_P1_PPO_Continuous.ipynb

# Run SAC/DDPG comparison
jupyter notebook code/HW4_P2_SAC_DDPG_Continuous.ipynb
```

## 📈 Expected Results

### PPO on Pendulum
- Converges in ~100-200 episodes
- Final performance: -200 to -150 reward
- Stable training with low variance

### SAC vs DDPG on HalfCheetah
- **SAC**: ~1M steps to 5000+ reward
- **DDPG**: ~1.5M steps, less stable
- **SAC**: Better exploration, more robust

## 🐛 Common Issues & Solutions

### Issue 1: PPO Not Learning
**Solutions:**
- Check advantage normalization
- Ensure proper GAE computation
- Verify clipping ratio (try 0.1-0.3)
- Increase epochs or batch size

### Issue 2: DDPG Unstable
**Solutions:**
- Reduce learning rate
- Increase target update τ (try 0.005)
- Add gradient clipping
- Tune exploration noise

### Issue 3: SAC Overexplores
**Solutions:**
- Check target entropy (default: -dim(action))
- Monitor α value
- Ensure proper reward scaling

## 📖 References

### Key Papers

1. **Schulman, J., et al. (2017)** - "Proximal Policy Optimization Algorithms" [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

2. **Lillicrap, T. P., et al. (2015)** - "Continuous control with deep reinforcement learning" [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

3. **Haarnoja, T., et al. (2018)** - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

4. **Schulman, J., et al. (2016)** - "High-Dimensional Continuous Control Using Generalized Advantage Estimation" [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

## 💡 Discussion Questions

1. Why does PPO outperform vanilla policy gradients?
2. How does SAC's entropy term improve exploration?
3. Why use twin Q-networks in SAC?
4. When would deterministic policies (DDPG) be preferred over stochastic (SAC)?
5. What are the trade-offs between on-policy (PPO) and off-policy (SAC) methods?

## 🎓 Extensions

- Implement TD3 (Twin Delayed DDPG)
- Add hindsight experience replay
- Try multi-task learning
- Implement distributed PPO (DPPO/IMPALA)

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024
