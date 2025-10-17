# HW11: Meta-Learning in Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Meta-Learning](https://img.shields.io/badge/Type-Meta--Learning-purple.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

Meta-learning, or "learning to learn," enables agents to quickly adapt to new tasks by leveraging experience from related tasks. This assignment explores meta-RL algorithms that achieve few-shot adaptation and transfer learning across task distributions.

## 🎯 Learning Objectives

1. **Meta-Learning Fundamentals**: Understand the paradigm of learning to learn
2. **MAML for RL**: Master model-agnostic meta-learning in RL settings
3. **Few-Shot Adaptation**: Learn to solve new tasks with minimal samples
4. **Task Distributions**: Handle families of related RL tasks
5. **Context-Based Meta-RL**: Use recurrent networks to encode task information
6. **Applications**: Transfer learning, sim-to-real, personalization

## 📂 Directory Structure

```
HW11_Meta_Learning/
├── code/
│   └── HW11_Notebook.ipynb        # Meta-RL implementations
├── answers/                        # (No solutions provided yet)
├── reports/
│   └── HW11_Questions.pdf         # Assignment questions
└── README.md
```

## 📚 Core Concepts

### 1. Meta-Learning Problem Formulation

**Goal:** Learn across tasks to enable fast adaptation to new tasks

**Setup:**
```
Task distribution: p(T)
Each task Ti has:
- State space Si
- Action space Ai  
- Reward function Ri
- Dynamics Pi

Meta-Training: Sample tasks from p(T), learn meta-policy
Meta-Testing: Adapt quickly to new task from p(T)
```

**Two-Level Optimization:**
```
Inner Loop: Adapt to specific task Ti
  θ_i ← θ - α∇θ L_Ti(θ)

Outer Loop: Optimize for fast adaptation
  θ ← θ - β∇θ ∑i L_Ti(θ_i)
```

### 2. Model-Agnostic Meta-Learning (MAML)

**Key Idea:** Find initialization that is good for fine-tuning

**Algorithm:**
```python
def maml(task_distribution, meta_lr=0.001, inner_lr=0.01, K=5):
    """
    MAML for RL
    
    Args:
        task_distribution: Distribution over tasks
        meta_lr: Outer loop learning rate
        inner_lr: Inner loop learning rate  
        K: Number of inner gradient steps
    """
    # Initialize meta-parameters
    theta = initialize_policy()
    
    for meta_iteration in range(N_meta):
        # Sample batch of tasks
        tasks = task_distribution.sample(batch_size)
        
        meta_gradient = 0
        
        for task in tasks:
            # Copy current parameters
            phi = theta.clone()
            
            # INNER LOOP: Adapt to task
            for k in range(K):
                # Sample trajectories using phi
                trajectories = collect_trajectories(task, phi)
                
                # Compute task loss (RL objective)
                loss = compute_rl_loss(trajectories, phi)
                
                # Inner update
                grad = compute_gradient(loss, phi)
                phi = phi - inner_lr * grad
            
            # OUTER LOOP: Meta-gradient
            # Sample new trajectories with adapted policy
            test_trajectories = collect_trajectories(task, phi)
            meta_loss = compute_rl_loss(test_trajectories, phi)
            
            # Compute gradient w.r.t. theta (not phi!)
            meta_grad = compute_gradient(meta_loss, theta)
            meta_gradient += meta_grad
        
        # Meta-update
        theta = theta - meta_lr * meta_gradient / batch_size
    
    return theta
```

**Key Innovation:** Backpropagation through optimization process

**For RL:**
```python
class MAML_RL:
    def compute_rl_loss(self, trajectories, policy):
        """
        Can use any RL objective:
        - Policy gradient
        - PPO objective
        - Value function loss
        """
        states, actions, returns, advantages = process(trajectories)
        
        # Policy gradient loss
        log_probs = policy.log_prob(states, actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value function loss
        values = policy.value(states)
        value_loss = F.mse_loss(values, returns)
        
        return policy_loss + 0.5 * value_loss
```

**Challenges in RL:**
- High variance in gradients
- Need multiple rollouts per task
- Computationally expensive
- Second-order derivatives

**Solutions:**
- Use more samples per task
- First-order approximation (FOMAML)
- Larger inner learning rate
- PPO for stability

### 3. Recurrent Meta-RL (RL²)

**Key Idea:** Use recurrent network to encode task

**No explicit adaptation:** Network learns to adapt through hidden state

```python
class RL2(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Recurrent encoder
        self.lstm = nn.LSTM(
            input_size=obs_dim + action_dim + 1 + 1,  # obs + prev_action + prev_reward + done
            hidden_size=hidden_dim,
            num_layers=2
        )
        
        # Policy head
        self.policy = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, prev_action, prev_reward, done, hidden):
        """
        Args:
            obs: Current observation
            prev_action: Previous action taken
            prev_reward: Previous reward received
            done: Episode termination flag
            hidden: LSTM hidden state
        """
        # Concatenate context
        x = torch.cat([obs, prev_action, prev_reward.unsqueeze(-1), done.unsqueeze(-1)], dim=-1)
        
        # Update hidden state (encodes task)
        output, hidden_new = self.lstm(x.unsqueeze(0), hidden)
        
        # Compute policy and value
        policy_logits = self.policy(output.squeeze(0))
        value = self.value(output.squeeze(0))
        
        return policy_logits, value, hidden_new
    
    def reset_hidden(self, batch_size=1):
        """Reset hidden state for new task"""
        return (torch.zeros(2, batch_size, 256),
                torch.zeros(2, batch_size, 256))
```

**Training:**
```python
def train_rl2(meta_tasks, episodes_per_task=10):
    """
    Train RL² across tasks
    Each "episode" is actually multiple episodes from same task
    """
    policy = RL2(obs_dim, action_dim)
    
    for meta_iteration in range(N):
        task = meta_tasks.sample()
        
        # Initialize hidden state
        hidden = policy.reset_hidden()
        
        # Collect multiple episodes from same task
        trajectories = []
        for episode in range(episodes_per_task):
            trajectory, hidden = collect_episode(
                task, policy, hidden, reset_hidden=False
            )
            trajectories.append(trajectory)
        
        # Train with PPO on all trajectories
        loss = compute_ppo_loss(trajectories, policy)
        optimize(loss)
```

**Advantages:**
- No explicit inner loop
- Fast adaptation at test time (just forward pass)
- Can handle varying task horizons

**Disadvantages:**
- Requires many episodes per task during meta-training
- Limited by LSTM capacity
- Black-box adaptation

### 4. Context-Based Meta-RL

**Key Idea:** Learn task embedding, condition policy on it

```python
class PEARL(nn.Module):
    """
    Probabilistic Embeddings for Actor-critic RL
    """
    def __init__(self, obs_dim, action_dim, context_dim):
        super().__init__()
        
        # Context encoder (variational)
        self.context_encoder = nn.LSTM(
            input_size=obs_dim + action_dim + 1,  # obs + action + reward
            hidden_size=128
        )
        
        self.context_mean = nn.Linear(128, context_dim)
        self.context_logstd = nn.Linear(128, context_dim)
        
        # Policy conditioned on context
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Q-function conditioned on context
        self.q_function = nn.Sequential(
            nn.Linear(obs_dim + action_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def encode_context(self, transitions):
        """
        Encode task from transitions
        Returns: mean and std of context distribution
        """
        # transitions: [(s, a, r, s'), ...]
        inputs = torch.cat([s, a, r.unsqueeze(-1)], dim=-1)
        
        output, _ = self.context_encoder(inputs)
        pooled = output.mean(dim=0)  # Aggregate over transitions
        
        mean = self.context_mean(pooled)
        logstd = self.context_logstd(pooled)
        
        return mean, logstd
    
    def sample_context(self, mean, logstd):
        """Sample context vector"""
        std = torch.exp(logstd)
        return mean + std * torch.randn_like(std)
    
    def forward(self, obs, context):
        """
        Action selection conditioned on context
        """
        x = torch.cat([obs, context], dim=-1)
        return self.policy(x)
```

**Meta-Training:**
```python
def meta_train_pearl(task_distribution):
    model = PEARL(obs_dim, action_dim, context_dim)
    
    for iteration in range(N):
        task = task_distribution.sample()
        
        # Phase 1: Collect context (few transitions)
        context_transitions = collect_transitions(task, model, n=10)
        
        # Encode task
        mean, logstd = model.encode_context(context_transitions)
        context = model.sample_context(mean, logstd)
        
        # Phase 2: Collect more data with inferred context
        trajectories = collect_trajectories(task, model, context, n=100)
        
        # Update model (SAC + context encoder)
        sac_loss = compute_sac_loss(trajectories, context)
        
        # Context encoder loss (variational)
        kl_loss = compute_kl(mean, logstd)
        
        total_loss = sac_loss + beta * kl_loss
        optimize(total_loss)
```

**Meta-Testing (Few-Shot Adaptation):**
```python
def adapt(new_task, model, n_context=5):
    """
    Adapt to new task with n_context transitions
    """
    # Collect minimal context
    context_data = collect_transitions(new_task, model, n=n_context)
    
    # Infer task embedding
    mean, logstd = model.encode_context(context_data)
    context = mean  # Use mean (no sampling) at test time
    
    # Now can act on new task
    def policy(obs):
        return model(obs, context)
    
    return policy
```

### 5. Meta-World Benchmark

**Standard benchmark for meta-RL:**
- 50 robotic manipulation tasks
- Shared state/action spaces
- Varying goals and object positions
- Evaluation: success rate after K adaptations steps

**Tasks include:**
- Reach, push, pick-place
- Drawer opening, button pressing
- Door unlocking, window closing

### 6. Key Challenges

1. **Sample Efficiency**
   - Need many tasks for meta-training
   - Each task requires episodes
   - Total sample cost high

2. **Task Distribution**
   - Performance depends on task similarity
   - Need diverse but related tasks
   - Distribution shift at test time problematic

3. **Computational Cost**
   - MAML requires second-order derivatives
   - Multiple rollouts per task
   - Slow meta-training

4. **Generalization**
   - Overfitting to meta-training tasks
   - Extrapolation vs interpolation
   - Out-of-distribution tasks

## 📊 Topics Covered

1. **MAML**: Gradient-based meta-learning
2. **RL²**: Recurrent meta-RL
3. **PEARL**: Context-based meta-RL
4. **Task Embeddings**: Learning task representations
5. **Few-Shot RL**: Rapid adaptation
6. **Meta-World**: Benchmark tasks

## 📖 Key References

1. **Finn, C., et al. (2017)** - "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" - ICML

2. **Duan, Y., et al. (2016)** - "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning" - arXiv

3. **Rakelly, K., et al. (2019)** - "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables" - ICML (PEARL)

4. **Yu, T., et al. (2020)** - "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning" - CoRL

## 💡 Discussion Questions

1. How does MAML differ from standard transfer learning?
2. Why is RL² considered "black-box" meta-learning?
3. What are trade-offs between gradient-based and recurrent meta-RL?
4. How does context-based meta-RL achieve fast adaptation?
5. When is meta-learning preferable to multi-task learning?

## 🎓 Extensions

- Implement first-order MAML (FOMAML)
- Try meta-learning with world models
- Explore meta-imitation learning
- Study meta-learning for exploration
- Apply to sim-to-real transfer

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024
