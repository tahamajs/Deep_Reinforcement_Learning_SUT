# HW9: Advanced RL Algorithms

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Advanced](https://img.shields.io/badge/Level-Advanced-red.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment covers state-of-the-art reinforcement learning algorithms that push the boundaries of performance, sample efficiency, and stability. Topics include distributional RL, Rainbow DQN, TD3, advanced policy optimization, and recent breakthroughs.

## 🎯 Learning Objectives

1. **Distributional RL**: Learn to model full return distributions, not just expectations
2. **Rainbow DQN**: Understand how to combine multiple DQN improvements
3. **Twin Delayed DDPG (TD3)**: Master advanced continuous control techniques
4. **Advanced Policy Optimization**: Study TRPO, PPO variants, and natural gradients
5. **Value Function Estimation**: Learn double Q-learning, dueling architectures
6. **State-of-the-Art Methods**: Understand current best practices in deep RL

## 📚 Core Concepts

### 1. Distributional Reinforcement Learning

**Key Insight:** Model the full distribution of returns, not just the expectation.

**Why Distributions Matter:**
```
Two slot machines:
A: Always gives $10          → E[A] = $10, Var = 0
B: 50% $0, 50% $20          → E[B] = $10, Var = 100

Expected value is same, but risk profiles differ!
```

**Distributional Bellman Equation:**
```
Z(s,a) ≐ R(s,a) + γZ(s',a')  (random variable)

Instead of: Q(s,a) = E[R + γQ(s',a')]
```

#### C51: Categorical DQN
```python
class C51(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms)
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * num_atoms)
        )
    
    def forward(self, state):
        logits = self.network(state)
        logits = logits.view(-1, self.action_dim, self.num_atoms)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_q_values(self, state):
        probs = self.forward(state)
        q_values = (probs * self.support).sum(dim=-1)
        return q_values
```

**Projection Step (Key Innovation):**
```python
def project_distribution(next_dist, rewards, dones, gamma):
    """
    Project T_z (distributional Bellman) onto support
    """
    batch_size = rewards.shape[0]
    
    # Compute projected values: r + γ * support
    proj_support = rewards.unsqueeze(-1) + \
                   gamma * (1 - dones.unsqueeze(-1)) * support
    
    proj_support = proj_support.clamp(v_min, v_max)
    
    # Map to categorical distribution
    b = (proj_support - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    
    # Distribute probability
    projected_dist = torch.zeros_like(next_dist)
    for i in range(num_atoms):
        projected_dist[:, l[:, i]] += next_dist[:, i] * (u[:, i] - b[:, i])
        projected_dist[:, u[:, i]] += next_dist[:, i] * (b[:, i] - l[:, i])
    
    return projected_dist
```

**Benefits:**
- Richer value representations
- Better handles multi-modal returns
- Improved stability
- Better performance empirically

#### Quantile Regression DQN (QR-DQN)
Instead of fixed support, learn quantiles:
```
τ ∈ [0,1]: quantile level
Z_θ(s,a,τ): τ-quantile of return distribution

Loss: Quantile Huber Loss
```

### 2. Rainbow DQN: Combining Improvements

**Six Extensions Combined:**

1. **Double Q-Learning**: Reduce overestimation
2. **Prioritized Experience Replay**: Sample important transitions more
3. **Dueling Networks**: Separate value and advantage
4. **Multi-Step Returns**: Use n-step bootstrapping
5. **Distributional RL**: Model return distributions (C51)
6. **Noisy Nets**: Parameter space exploration

**Architecture:**
```python
class RainbowDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms, num_steps):
        super().__init__()
        
        # Feature extraction with noisy layers
        self.feature = nn.Sequential(
            NoisyLinear(state_dim, 128),
            nn.ReLU()
        )
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, num_atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim * num_atoms)
        )
        
        # Distributional RL
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms)
    
    def forward(self, state):
        features = self.feature(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        advantage = advantage.view(-1, self.action_dim, self.num_atoms)
        
        # Dueling combination
        q_atoms = value.unsqueeze(1) + \
                  (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Softmax for probability distribution
        q_dist = F.softmax(q_atoms, dim=-1)
        
        return q_dist
```

**Training with Prioritized Replay:**
```python
def train_rainbow(batch, priorities):
    states, actions, rewards, next_states, dones = batch
    
    # Multi-step returns (n-step)
    n_step_rewards = compute_n_step_returns(rewards, gamma, n_steps)
    
    # Distributional Bellman update
    current_dist = model(states)[range(batch_size), actions]
    
    with torch.no_grad():
        # Double Q-learning: use online net for action selection
        next_actions = model.get_q_values(next_states).argmax(dim=1)
        
        # Target net for evaluation
        next_dist = target_model(next_states)[range(batch_size), next_actions]
        
        # Project distribution
        target_dist = project_distribution(next_dist, n_step_rewards, dones)
    
    # Cross-entropy loss
    loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(dim=-1)
    
    # Importance sampling weights for prioritized replay
    loss = (loss * is_weights).mean()
    
    # Update priorities
    priorities = loss.detach()
    
    return loss, priorities
```

**Ablation Studies Show:**
- Each component contributes
- Prioritized replay + multi-step most important
- Distributional RL adds significant gain
- Combined: state-of-the-art on Atari

### 3. Twin Delayed DDPG (TD3)

**Motivation:** Address overestimation and brittleness in DDPG

**Three Key Ideas:**

#### 1. Twin Q-Networks (Clipped Double Q-Learning)
```python
Q1, Q2 = twin_critics(state, action)
Q_target = min(Q1, Q2)  # Take minimum to reduce overestimation
```

#### 2. Delayed Policy Updates
```python
if step % policy_delay == 0:
    update_actor()  # Update policy less frequently
update_critics()  # Always update critics
```

**Why:** Reduce variance in policy gradient from Q-function errors

#### 3. Target Policy Smoothing
```python
def target_policy_smoothing(next_state, epsilon=0.2, c=0.5):
    action = target_actor(next_state)
    noise = torch.randn_like(action) * epsilon
    noise = noise.clamp(-c, c)
    action = (action + noise).clamp(-1, 1)
    return action
```

**Why:** Make Q-function robust to small action perturbations

**Complete Algorithm:**
```python
# Critic update
a_next = target_actor(s_next) + noise.clamp(-c, c)
a_next = a_next.clamp(-1, 1)

Q1_next = target_critic1(s_next, a_next)
Q2_next = target_critic2(s_next, a_next)
Q_next = min(Q1_next, Q2_next)

y = r + γ * (1 - done) * Q_next

loss_Q = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

# Delayed actor update
if t % d == 0:
    loss_actor = -Q1(s, actor(s)).mean()
    
    # Soft update targets
    target_critic1 ← τ*critic1 + (1-τ)*target_critic1
    target_critic2 ← τ*critic2 + (1-τ)*target_critic2
    target_actor ← τ*actor + (1-τ)*target_actor
```

**Results:**
- More stable than DDPG
- Better final performance
- Less sensitive to hyperparameters

### 4. Trust Region Policy Optimization (TRPO)

**Key Idea:** Constrain policy updates to trust region

**Objective:**
```
maximize  𝔼[πθ(a|s)/πθ_old(a|s) * A(s,a)]
   θ

subject to: KL(πθ_old || πθ) ≤ δ
```

**Why Constrain:**
- Large policy changes can be catastrophic
- Trust region ensures monotonic improvement
- Theoretically motivated

**Natural Policy Gradient:**
```
∇θ J(θ) ≈ F(θ)⁻¹ ∇θ L(θ)

where F(θ) is Fisher information matrix
```

**Implementation Challenge:** Computing F⁻¹ expensive

**Solution:** Conjugate Gradient

**Practical Algorithm (Simplified):**
1. Collect trajectories using πθ_old
2. Compute advantages
3. Compute natural gradient direction using conjugate gradient
4. Line search to satisfy KL constraint
5. Update policy

**Advantages:**
- Monotonic improvement guarantee
- Robust across tasks
- Good empirical performance

**Disadvantages:**
- Computationally expensive (conjugate gradient, line search)
- Complex to implement correctly

### 5. Advanced Value Function Techniques

#### Dueling Networks
```python
class DuelingDQN(nn.Module):
    def forward(self, state):
        features = self.feature_layer(state)
        
        # State value
        V = self.value_stream(features)
        
        # Action advantages
        A = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,.)))
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        
        return Q
```

**Benefit:** Better identify which states are valuable

#### Retrace(λ)
Off-policy multi-step returns with importance sampling correction:
```
R_t^λ = rt + γ * min(1, ρt) * [R_{t+1}^λ - Q(st+1, at+1)] + γ * Q(st+1, at+1)

where ρt = π(at|st) / μ(at|st)
```

**Benefits:**
- Safe off-policy learning
- Low variance
- Multi-step returns

## 📊 Topics Covered

1. **Distributional RL**: C51, QR-DQN, IQN
2. **Rainbow Components**: Each of 6 extensions
3. **TD3**: Twin critics, delayed updates, smoothing
4. **Trust Regions**: TRPO, natural gradients
5. **Advanced Architectures**: Dueling, attention, transformers
6. **Multi-Step Methods**: n-step, Retrace(λ)

## 📖 Key References

1. **Bellemare, M. G., et al. (2017)** - "A Distributional Perspective on RL" - ICML
2. **Hessel, M., et al. (2018)** - "Rainbow: Combining Improvements in Deep RL" - AAAI
3. **Fujimoto, S., et al. (2018)** - "Addressing Function Approximation Error in Actor-Critic Methods" - ICML (TD3)
4. **Schulman, J., et al. (2015)** - "Trust Region Policy Optimization" - ICML
5. **Wang, Z., et al. (2016)** - "Dueling Network Architectures" - ICML

## 💡 Discussion Questions

1. Why does modeling full distributions help in RL?
2. Which Rainbow components contribute most to performance?
3. How does TD3 improve upon DDPG?
4. What are trade-offs between PPO and TRPO?
5. When should you use distributional vs expectation-based methods?

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024
