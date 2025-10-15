# HW13: Offline Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Offline](https://img.shields.io/badge/Type-Offline-darkgreen.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

Offline Reinforcement Learning (also called Batch RL) learns policies from fixed datasets without environment interaction. This is crucial for real-world applications where online interaction is expensive, dangerous, or impossible. This assignment explores challenges unique to offline RL and state-of-the-art solutions.

## 🎯 Learning Objectives

1. **Offline RL Fundamentals**: Understand learning from fixed datasets
2. **Distributional Shift**: Master the core challenge of offline RL
3. **Conservative Methods**: Learn to avoid over-optimistic value estimates
4. **Behavior Regularization**: Constrain policies to stay near data distribution
5. **Practical Applications**: Healthcare, robotics, recommendation systems
6. **Evaluation**: Understand unique challenges in offline evaluation

## 📂 Directory Structure

```
HW13_Offline_RL/
├── code/
│   ├── HW_13_first_part.ipynb     # Offline RL algorithms (part 1)
│   └── HW_13_second_part.ipynb    # Advanced methods (part 2)
├── answers/                        # (No solutions provided yet)
├── reports/
│   └── HW13_Questions.pdf         # Assignment questions
└── README.md
```

## 📚 Core Concepts

### 1. Offline RL Problem Setting

**Setup:**
```
Given: Fixed dataset D = {(s, a, r, s')}
       Collected by behavior policy π_β (unknown)
       
Goal:  Learn policy π that maximizes:
       J(π) = 𝔼_{s~ρ, a~π}[∑ γᵗ r(st, at)]
       
Constraint: NO environment interaction during training
```

**Why Offline RL?**

**Applications:**
- **Healthcare**: Learn from historical patient data
- **Robotics**: Avoid dangerous exploration on real robots
- **Recommendations**: Use logged user interactions
- **Autonomous Driving**: Learn from human driving data
- **Finance**: Learn from historical trading data

**Key Difference from Online RL:**
```
Online RL:  Policy → Collect Data → Improve Policy → Repeat
Offline RL: Fixed Data → Learn Policy (no interaction)
```

### 2. The Distributional Shift Problem

**Core Challenge:** Querying Q(s,a) where (s,a) not in dataset

**Extrapolation Error:**
```python
# Dataset collected by behavior policy
D = {(s, a, r, s') ~ π_β}

# Learned policy may choose OOD actions
π_learned(s) = argmax Q(s, a)
                 a
                 
# If π_learned(s) ∉ support(π_β(s)):
#   → Q(s, π_learned(s)) is extrapolated
#   → May be wildly inaccurate!
#   → Policy exploits errors
```

**Illustration:**
```
Q-values fitted to data (dots):

 Q  |     
    |   *   *
    | *   ?   *    ← ? = OOD action, Q may be wrong
    |   *   *
    |_____________
       actions

Policy picks argmax → selects OOD action with overestimated Q
```

**Why It's Catastrophic:**
```
1. Policy chooses action with highest Q-value
2. If Q-value overestimated for OOD action → policy selects it
3. No correction (can't interact to get true reward)
4. Cascading errors through Bellman backup
```

### 3. Conservative Q-Learning (CQL)

**Key Idea:** Learn Q-function that lower-bounds true Q-values

**Objective:**
```
minimize: 𝔼_{s~D}[log ∑_a exp(Q(s,a))] - 𝔼_{(s,a)~D}[Q(s,a)]
   Q                ↑                           ↑
              Pushes down Q        Don't underestimate data

Subject to: Bellman error on dataset D
```

**Intuition:**
- First term: Minimize Q-values for all actions
- Second term: Don't underestimate actions in dataset
- Result: Conservative (lower-bound) Q-function

**Implementation:**
```python
class CQL:
    def __init__(self, state_dim, action_dim, alpha=1.0):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)
        self.policy = Policy(state_dim, action_dim)
        self.alpha = alpha  # CQL regularization strength
    
    def cql_loss(self, states, actions, rewards, next_states):
        # Standard SAC losses
        q1_pred = self.Q1(states, actions)
        q2_pred = self.Q2(states, actions)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q_target = rewards + gamma * (
                torch.min(
                    self.Q1_target(next_states, next_actions),
                    self.Q2_target(next_states, next_actions)
                ) - alpha_entropy * next_log_probs
            )
        
        bellman_error = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        
        # CQL regularization
        # Sample actions for CQL penalty
        random_actions = torch.rand(states.shape[0], action_dim) * 2 - 1
        current_actions, current_log_probs = self.policy.sample(states)
        
        # Q-values for different action distributions
        q1_rand = self.Q1(states, random_actions)
        q1_curr = self.Q1(states, current_actions)
        q1_data = self.Q1(states, actions)
        
        # Logsumexp of Q-values (pushes all Q down)
        cql_q1_loss = torch.logsumexp(
            torch.cat([q1_rand, q1_curr], dim=1), dim=1
        ).mean()
        
        # Don't underestimate data Q-values
        cql_q1_loss -= q1_data.mean()
        
        # Same for Q2
        cql_q2_loss = # ... similar
        
        # Total Q-loss
        q_loss = bellman_error + self.alpha * (cql_q1_loss + cql_q2_loss)
        
        return q_loss
    
    def train_step(self, batch):
        # Update Q-functions with CQL
        q_loss = self.cql_loss(*batch)
        optimize_q(q_loss)
        
        # Update policy (standard SAC)
        policy_loss = self.policy_loss(*batch)
        optimize_policy(policy_loss)
```

**Why CQL Works:**
- Prevents overestimation on OOD actions
- Bounded policy improvement
- Strong empirical results across domains

### 4. Implicit Q-Learning (IQL)

**Key Idea:** Avoid explicit maximization in Bellman backup

**Problem with Standard Q-Learning:**
```
Q(s,a) = r + γ max Q(s', a')  ← max causes extrapolation error
                a'
```

**IQL Solution:** Use expectile regression
```
V(s) ≈ 𝔼_π[Q(s,a)]  (no max!)

Q(s,a) = r + γV(s')  (use V, not max Q)
```

**Expectile Regression:**
```python
def expectile_loss(pred, target, expectile=0.7):
    """
    Asymmetric squared loss
    expectile > 0.5 → emphasizes upper tail
    """
    errors = target - pred
    weight = torch.where(
        errors > 0,
        expectile,
        1 - expectile
    )
    return (weight * errors**2).mean()
```

**Algorithm:**
```python
class IQL:
    def __init__(self, state_dim, action_dim, expectile=0.7):
        self.Q = QNetwork(state_dim, action_dim)
        self.V = VNetwork(state_dim)
        self.policy = Policy(state_dim, action_dim)
        self.expectile = expectile
    
    def train_step(self, states, actions, rewards, next_states):
        # 1. Update V using expectile regression
        with torch.no_grad():
            q_values = self.Q(states, actions)
        
        v_pred = self.V(states)
        v_loss = expectile_loss(v_pred, q_values, self.expectile)
        
        # 2. Update Q using V (not max Q!)
        with torch.no_grad():
            v_next = self.V(next_states)
            q_target = rewards + gamma * v_next
        
        q_pred = self.Q(states, actions)
        q_loss = F.mse_loss(q_pred, q_target)
        
        # 3. Update policy with advantage-weighted regression
        with torch.no_grad():
            advantage = q_pred - v_pred
            weights = torch.exp(advantage / beta).clamp(max=100)
        
        log_probs = self.policy.log_prob(states, actions)
        policy_loss = -(weights * log_probs).mean()
        
        return v_loss, q_loss, policy_loss
```

**Advantages:**
- No explicit max → less extrapolation error
- Simple to implement
- Comparable to CQL but faster
- Works well with suboptimal data

### 5. Behavior Regularization

**Key Idea:** Constrain learned policy to stay near behavior policy

#### Batch-Constrained Q-Learning (BCQ)

**For Discrete Actions:**
```python
class BCQ:
    def __init__(self, state_dim, action_dim):
        self.Q = QNetwork(state_dim, action_dim)
        self.G = GenerativeModel(state_dim, action_dim)  # Models π_β
        self.perturbation = PerturbationNetwork(state_dim, action_dim)
    
    def select_action(self, state, epsilon=0.3):
        # Sample actions from behavior model
        actions = self.G.sample(state, n=10)
        
        # Perturb slightly
        perturbed = actions + self.perturbation(state, actions)
        
        # Select based on Q-values
        q_values = self.Q(state, perturbed)
        best_idx = q_values.argmax()
        
        return perturbed[best_idx]
```

#### AWR (Advantage-Weighted Regression)
```python
def awr_policy_update(states, actions, advantages, policy, beta=0.05):
    """
    Update policy with advantage-weighted regression
    beta: temperature parameter
    """
    # Weight actions by exponentiated advantage
    weights = torch.exp(advantages / beta)
    weights = weights.clamp(max=20)  # Clip for stability
    
    # Supervised learning weighted by advantage
    log_probs = policy.log_prob(states, actions)
    loss = -(weights.detach() * log_probs).mean()
    
    return loss
```

### 6. Model-Based Offline RL (MOPO)

**Key Idea:** Learn dynamics model, use it carefully for planning

**Conservative Model Usage:**
```python
class MOPO:
    def __init__(self, ensemble_size=7):
        # Ensemble of dynamics models
        self.dynamics = [DynamicsModel() for _ in range(ensemble_size)]
        self.policy = SACPolicy()
    
    def model_rollout(self, state, action):
        """
        Rollout with uncertainty penalty
        """
        # Predict next state with each model
        predictions = [model(state, action) for model in self.dynamics]
        
        # Mean prediction
        next_state_mean = torch.stack([p[0] for p in predictions]).mean(dim=0)
        reward_mean = torch.stack([p[1] for p in predictions]).mean(dim=0)
        
        # Epistemic uncertainty (model disagreement)
        next_state_std = torch.stack([p[0] for p in predictions]).std(dim=0)
        
        # Penalize high uncertainty
        penalty = lambda_u * next_state_std.mean()
        adjusted_reward = reward_mean - penalty
        
        return next_state_mean, adjusted_reward
    
    def train(self, dataset):
        # 1. Train ensemble on dataset
        for model in self.dynamics:
            train_dynamics(model, dataset)
        
        # 2. Generate synthetic data with penalty
        synthetic_data = []
        for state, action in dataset:
            next_state, reward = self.model_rollout(state, action)
            synthetic_data.append((state, action, reward, next_state))
        
        # 3. Train policy on real + synthetic data
        combined_data = dataset + synthetic_data
        train_policy_offline(self.policy, combined_data)
```

**Why It Works:**
- Uses model to generate more data
- Uncertainty penalty keeps agent in-distribution
- Combines benefits of model-based and model-free

### 7. Offline RL Datasets

**D4RL Benchmark:**
- **MuJoCo**: Locomotion tasks (walker, hopper, etc.)
- **Adroit**: Robotic manipulation
- **AntMaze**: Navigation with sparse rewards
- **Flow**: Traffic control

**Dataset Quality Levels:**
- **Random**: Uniformly random actions
- **Medium**: Partially trained policy
- **Medium-Replay**: All data from training online agent
- **Expert**: Fully trained policy
- **Medium-Expert**: Mix of medium and expert

### 8. Evaluation Challenges

**Problems:**
- Can't measure true performance (no environment)
- Off-policy evaluation is biased
- Need to be careful with learned models

**Solutions:**
- **Hold-out evaluation**: If environment available
- **Model-based evaluation**: Learn model, simulate
- **Off-policy evaluation**: Importance sampling, FQE
- **Uncertainty estimates**: Report confidence intervals

## 📊 Topics Covered

1. **Distributional Shift**: Core offline RL challenge
2. **CQL**: Conservative value estimation
3. **IQL**: Implicit Q-learning with expectiles
4. **BCQ**: Behavior-constrained learning
5. **MOPO**: Model-based with uncertainty
6. **Evaluation**: Off-policy evaluation methods

## 📖 Key References

1. **Levine, S., et al. (2020)** - "Offline Reinforcement Learning: Tutorial, Review, and Perspectives" - arXiv:2005.01643

2. **Kumar, A., et al. (2020)** - "Conservative Q-Learning for Offline RL" - NeurIPS (CQL)

3. **Kostrikov, I., et al. (2021)** - "Offline Reinforcement Learning with Implicit Q-Learning" - arXiv (IQL)

4. **Fujimoto, S., et al. (2019)** - "Off-Policy Deep RL without Exploration" - ICML (BCQ)

5. **Yu, T., et al. (2020)** - "MOPO: Model-based Offline Policy Optimization" - NeurIPS

6. **Fu, J., et al. (2021)** - "D4RL: Datasets for Deep Data-Driven RL" - arXiv

## 💡 Discussion Questions

1. Why is distributional shift more severe in offline RL than online?
2. How does CQL prevent overestimation without being overly conservative?
3. What are trade-offs between behavior regularization and value regularization?
4. When would model-based offline RL be preferred?
5. How can we evaluate offline RL algorithms without environment access?

## 🎓 Extensions

- Implement offline-to-online fine-tuning
- Try decision transformers (offline RL as sequence modeling)
- Explore offline RL for NLP/recommendation
- Study pessimism vs conservatism trade-offs
- Apply to real-world logged datasets

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024
