# HW6: Advanced Topics in Deep RL

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Advanced](https://img.shields.io/badge/Level-Advanced-red.svg)](https://www.deepmind.com/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment covers advanced topics and extensions in deep reinforcement learning, including curiosity-driven learning, intrinsic motivation, meta-learning, and multi-task RL. These techniques address fundamental challenges such as sparse rewards, exploration, and generalization.

## 🎯 Learning Objectives

1. **Intrinsic Motivation**: Learn curiosity-driven exploration methods
2. **Reward Shaping**: Understand how to design auxiliary rewards
3. **Transfer Learning**: Apply knowledge across related tasks
4. **Multi-Task RL**: Train single agent for multiple tasks
5. **Meta-Learning**: Learn to learn quickly from few samples
6. **Exploration Strategies**: Master advanced exploration techniques

## 📂 Directory Structure

```
HW6_Advanced_Topics/
├── code/
│   └── HW6_Notebook.ipynb               # All advanced topics
├── answers/
│   ├── HW6_Notebook_Solution.ipynb
│   └── HW6_Solution.pdf
├── reports/
│   └── HW6_Questions.pdf
└── README.md
```

## 📚 Core Concepts

### 1. Curiosity-Driven Learning

**Problem:** Sparse rewards make exploration difficult.

**Solution:** Add intrinsic reward based on novelty or prediction error.

**Intrinsic Curiosity Module (ICM):**
```
Intrinsic Reward = η || f̂(st, at) - st+1 ||²

where f̂ is learned forward model
```

**Types of Curiosity:**
- **Forward Model Error**: Prediction error of next state
- **Random Network Distillation (RND)**: Prediction error of random target network
- **Disagreement**: Variance among ensemble predictions
- **Information Gain**: Reduction in uncertainty

**Implementation:**
```python
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Forward model: predict next state features
        self.forward_model = nn.Sequential(
            nn.Linear(64 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Inverse model: predict action from states
        self.inverse_model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state, action, next_state):
        # Encode states
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)
        
        # Forward model loss
        phi_s_next_pred = self.forward_model(
            torch.cat([phi_s, action], dim=-1)
        )
        forward_loss = F.mse_loss(phi_s_next_pred, phi_s_next.detach())
        
        # Inverse model loss
        action_pred = self.inverse_model(
            torch.cat([phi_s, phi_s_next], dim=-1)
        )
        inverse_loss = F.cross_entropy(action_pred, action)
        
        # Intrinsic reward
        intrinsic_reward = forward_loss.detach()
        
        return intrinsic_reward, forward_loss, inverse_loss
```

### 2. Random Network Distillation (RND)

**Key Idea:** Use prediction error of random network as exploration bonus.

**Why It Works:**
- Frequently visited states → predictor learns well → low error → low bonus
- Novel states → high prediction error → high bonus → exploration

**Algorithm:**
```python
class RND(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # Fixed random target network
        self.target = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Trainable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, state):
        target_features = self.target(state)
        predicted_features = self.predictor(state)
        
        # Intrinsic reward = prediction error
        intrinsic_reward = F.mse_loss(
            predicted_features, 
            target_features.detach(),
            reduction='none'
        ).mean(dim=-1)
        
        return intrinsic_reward
```

**Advantages:**
- No dynamics model needed
- Computationally efficient
- Works in high-dimensional spaces
- Used in Montezuma's Revenge success

### 3. Multi-Task Reinforcement Learning

**Goal:** Train single agent to perform multiple tasks.

**Approaches:**

#### a) Multi-Head Networks
```python
class MultiTaskPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_tasks):
        super().__init__()
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(256, action_dim) 
            for _ in range(num_tasks)
        ])
    
    def forward(self, state, task_id):
        features = self.shared(state)
        logits = self.heads[task_id](features)
        return F.softmax(logits, dim=-1)
```

#### b) Task Conditioning
```python
class TaskConditionedPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, task_embedding_dim):
        super().__init__()
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        
        # Conditioned policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim + task_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, state, task_id):
        task_emb = self.task_embedding(task_id)
        x = torch.cat([state, task_emb], dim=-1)
        return self.policy(x)
```

**Benefits:**
- Transfer learning across tasks
- Improved sample efficiency
- Better generalization
- Single model deployment

### 4. Meta-Reinforcement Learning

**Problem:** Can agent learn to learn? Quickly adapt to new tasks?

**Model-Agnostic Meta-Learning (MAML):**
```
Find initial parameters θ such that after K gradient steps
on new task Ti, performance is maximized:

θ* = argmin ∑i L_Ti(θ - α∇θL_Ti(θ))
       θ
```

**Algorithm:**
```python
def maml_meta_train(tasks, meta_lr=0.001, inner_lr=0.01, inner_steps=5):
    # Initialize meta-parameters
    meta_params = initialize_policy()
    
    for meta_iteration in range(N):
        meta_gradient = 0
        
        for task in sample_tasks(tasks):
            # Inner loop: adapt to task
            params = meta_params.clone()
            
            for k in range(inner_steps):
                # Sample batch from task
                batch = task.sample()
                
                # Compute task loss and gradient
                loss = compute_loss(params, batch)
                grad = torch.autograd.grad(loss, params)
                
                # Inner update
                params = params - inner_lr * grad
            
            # Outer loop: meta-gradient
            test_batch = task.sample()
            test_loss = compute_loss(params, test_batch)
            meta_grad = torch.autograd.grad(test_loss, meta_params)
            
            meta_gradient += meta_grad
        
        # Meta-update
        meta_params = meta_params - meta_lr * meta_gradient
    
    return meta_params
```

**Applications:**
- Few-shot RL: Learn from few samples in new task
- Fast adaptation to new environments
- Robotic manipulation with varied objects

### 5. Hierarchical Reinforcement Learning

**Key Idea:** Learn policies at multiple time scales.

**Options Framework:**
```
Option = (Initiation Set, Policy, Termination Condition)

Example: "Navigate to door" is an option composed of:
- Low-level actions (move forward, turn)
- Termination: reached door
```

**Feudal Networks:**
```
Manager: Sets goals for Worker
Worker: Achieves goals set by Manager

Manager reward: extrinsic environment reward
Worker reward: intrinsic reward for achieving sub-goals
```

## 💻 Implementation Tasks

1. **Curiosity-Driven Exploration**
   - Implement ICM or RND
   - Test on sparse reward environments
   - Compare with ε-greedy exploration

2. **Multi-Task Learning**
   - Train on multiple related tasks
   - Measure positive transfer
   - Compare multi-head vs task-conditioning

3. **Meta-Learning**
   - Implement simple MAML
   - Test on few-shot adaptation
   - Measure adaptation speed

4. **Advanced Exploration**
   - Compare different intrinsic rewards
   - Analyze exploration vs exploitation
   - Visualize state visitation

## 📊 Evaluation Metrics

1. **Exploration Coverage**: Unique states visited
2. **Sample Efficiency**: Performance vs samples
3. **Transfer Performance**: Improvement on new tasks
4. **Adaptation Speed**: Steps to solve new task
5. **Intrinsic Reward Statistics**: Track over training

## 🔧 Requirements

```python
numpy>=1.21.0
matplotlib>=3.4.0
torch>=2.0.0
gymnasium>=0.28.0
higher>=0.2.1  # For MAML
```

## 📖 Key References

1. **Pathak, D., et al. (2017)** - "Curiosity-driven Exploration by Self-supervised Prediction" - ICML
2. **Burda, Y., et al. (2018)** - "Exploration by Random Network Distillation" - ICLR
3. **Finn, C., et al. (2017)** - "Model-Agnostic Meta-Learning for Fast Adaptation" - ICML
4. **Caruana, R. (1997)** - "Multitask Learning" - Machine Learning

## 💡 Discussion Questions

1. Why does prediction error work as intrinsic motivation?
2. When can multi-task learning hurt single-task performance?
3. What makes a good auxiliary task for transfer learning?
4. How does meta-learning differ from transfer learning?

## 🎓 Extensions

- Implement hindsight experience replay (HER)
- Try successor representations
- Implement soft option learning
- Explore world models for planning

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024

