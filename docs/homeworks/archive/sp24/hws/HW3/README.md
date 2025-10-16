# HW3: On-Policy Methods - Policy Gradients

## 📋 Overview

This assignment focuses on on-policy methods, implementing policy gradient algorithms from REINFORCE to PPO, mastering the foundations of modern policy-based reinforcement learning.

## 📂 Contents

```
HW3/
├── SP24_RL_HW3/
│   ├── RL_HW3.pdf              # Assignment description
│   └── RL_HW3_On_Policy.ipynb # Implementation notebook
└── README.md                     # This file
```

## 🎯 Learning Objectives

- ✅ Understand policy gradient theorem
- ✅ Implement REINFORCE algorithm
- ✅ Build actor-critic methods (A2C)
- ✅ Implement PPO with clipping
- ✅ Master advantage estimation (GAE)
- ✅ Compare on-policy algorithms

## 📚 Part 1: REINFORCE

### Policy Gradient Theorem

**Objective:**

```
J(θ) = E_{τ~πθ}[R(τ)] = E[∑ᵗ₌₀ γᵗ rₜ]
```

**Gradient:**

```
∇θJ(θ) = E[∑ᵗ₌₀ ∇θ log πθ(aₜ|sₜ) Gₜ]
```

### REINFORCE Algorithm

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

    def select_action(self, state):
        probs = self(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

def reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99):
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()

        # Generate episode
        while not done:
            action, log_prob = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # Normalize returns (reduces variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        loss = -sum([log_prob * G for log_prob, G in zip(actions, returns)])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### REINFORCE with Baseline

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)

def reinforce_baseline(env, policy, value_net, num_episodes=1000):
    policy_optimizer = Adam(policy.parameters(), lr=1e-3)
    value_optimizer = Adam(value_net.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        # ... generate episode ...

        # Compute advantages
        values = value_net(torch.FloatTensor(states))
        advantages = returns - values.detach()

        # Policy loss
        policy_loss = -sum([log_prob * adv for log_prob, adv in zip(log_probs, advantages)])

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Update
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
```

## 📚 Part 2: Actor-Critic (A2C)

### Advantage Actor-Critic

```python
def a2c(env, policy, value_net, num_episodes=1000, gamma=0.99):
    policy_optimizer = Adam(policy.parameters(), lr=1e-3)
    value_optimizer = Adam(value_net.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state = env.reset()
        log_probs, values, rewards = [], [], []

        while not done:
            # Policy step
            action, log_prob = policy.select_action(state)
            value = value_net(state)

            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns = compute_returns(rewards, gamma)

        # Advantages
        advantages = returns - torch.stack(values).squeeze()

        # Policy loss
        policy_loss = -sum([lp * adv for lp, adv in zip(log_probs, advantages)])

        # Value loss
        value_loss = F.mse_loss(torch.stack(values).squeeze(), returns)

        # Combined update
        loss = policy_loss + 0.5 * value_loss

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()
        value_optimizer.step()
```

## 📚 Part 3: Proximal Policy Optimization (PPO)

### Generalized Advantage Estimation

```python
def compute_gae(rewards, values, next_value, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
        next_value = values[t]

    return advantages
```

### PPO with Clipping

```python
def ppo_update(policy, old_policy, states, actions, old_log_probs, advantages,
               epsilon=0.2, ppo_epochs=10):

    for epoch in range(ppo_epochs):
        # New log probs
        new_log_probs = policy.get_log_prob(states, actions)

        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Update
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
```

### Complete PPO Algorithm

```python
def ppo(env, policy, value_net, num_iterations=1000):
    policy_optimizer = Adam(policy.parameters(), lr=3e-4)
    value_optimizer = Adam(value_net.parameters(), lr=1e-3)

    for iteration in range(num_iterations):
        # Collect trajectories
        trajectories = collect_trajectories(env, policy, num_steps=2048)

        states = trajectories['states']
        actions = trajectories['actions']
        old_log_probs = trajectories['log_probs']
        rewards = trajectories['rewards']
        values = trajectories['values']

        # Compute advantages and returns
        advantages = compute_gae(rewards, values, gamma=0.99, lambda_=0.95)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for epoch in range(10):
            # Mini-batch updates
            for batch in get_batches(states, actions, old_log_probs, advantages, returns):
                # Policy update
                policy_loss = compute_ppo_loss(policy, batch, epsilon=0.2)

                # Value update
                value_pred = value_net(batch['states'])
                value_loss = F.mse_loss(value_pred, batch['returns'])

                # Combined update
                policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy_optimizer.step()

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
```

## 🔬 Experiments

### Required Experiments

1. **Algorithm Comparison**

   - REINFORCE vs REINFORCE+Baseline
   - A2C vs PPO
   - Learning curves

2. **Hyperparameter Sensitivity**

   - Learning rate: [1e-4, 3e-4, 1e-3]
   - PPO clip ε: [0.1, 0.2, 0.3]
   - GAE λ: [0.9, 0.95, 0.99]

3. **Ablation Studies**
   - PPO without clipping
   - Without baseline
   - Different advantage estimations

## 📊 Expected Results

**CartPole-v1:**

- REINFORCE: Solves in 500-1000 episodes
- A2C: Solves in 300-500 episodes
- PPO: Solves in 200-400 episodes

**LunarLander-v2:**

- Target score: 200+
- PPO typically best performer

## 💡 Implementation Tips

- ✅ Normalize advantages (crucial for stability)
- ✅ Use orthogonal initialization
- ✅ Clip gradients (max_norm=0.5)
- ✅ Add entropy bonus for exploration
- ✅ Use learning rate annealing
- ✅ Monitor KL divergence

## 📖 References

- **Williams (1992)** - REINFORCE
- **Mnih et al. (2016)** - A3C
- **Schulman et al. (2017)** - PPO
- **Schulman et al. (2015)** - GAE

## ⏱️ Time Estimate

- Part 1 (REINFORCE): 4-6 hours
- Part 2 (A2C): 4-6 hours
- Part 3 (PPO): 6-10 hours
- Analysis: 3-5 hours
- **Total**: 17-27 hours

---

**Difficulty**: ⭐⭐⭐⭐☆ (Challenging)  
**Prerequisites**: HW2, Policy gradients  
**Key Skills**: Policy optimization, variance reduction

