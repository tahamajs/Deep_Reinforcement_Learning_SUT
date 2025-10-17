# HW4: Soft Actor-Critic (SAC)

## 📋 Overview

This assignment implements Soft Actor-Critic (SAC), a state-of-the-art off-policy algorithm based on the maximum entropy reinforcement learning framework. SAC combines the sample efficiency of off-policy methods with the stability of on-policy algorithms.

## 📂 Contents

```
HW4/
├── SP23_RL_HW4/
│   ├── RL_HW4.pdf                        # Assignment description
│   └── RL_HW4_Soft_Actor_Critic.ipynb   # Implementation notebook
└── README.md                              # This file
```

## 🎯 Learning Objectives

- ✅ Understand maximum entropy reinforcement learning
- ✅ Implement SAC with automatic temperature tuning
- ✅ Master stochastic policy optimization
- ✅ Apply to challenging continuous control tasks
- ✅ Compare with PPO and DDPG

## 📚 Background

### Maximum Entropy RL

**Standard RL Objective:**

```
max E[∑ᵗ γᵗ r(sₜ, aₜ)]
```

**Maximum Entropy Objective:**

```
max E[∑ᵗ γᵗ (r(sₜ, aₜ) + α H(π(·|sₜ)))]
```

where H(π(·|s)) = -E[log π(a|s)] is the entropy.

**Benefits:**

- Encourages exploration naturally
- Improves robustness
- Prevents premature convergence
- Learns multimodal behaviors

### SAC Key Features

1. **Off-Policy**: Uses replay buffer for sample efficiency
2. **Actor-Critic**: Combines policy and value learning
3. **Stochastic Policy**: Uses reparameterization trick
4. **Entropy Regularization**: Automatic temperature tuning
5. **Twin Q-Networks**: Mitigates overestimation like TD3

## 🏗️ Architecture

### 1. Actor Network (Stochastic Policy)

```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Action bounds
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Sample with reparameterization

        # Apply tanh squashing
        action = torch.tanh(x_t)
        action = action * self.action_scale + self.action_bias

        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Correction for tanh squashing
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean
```

**Key Points:**

- Gaussian policy with learned mean and std
- Reparameterization trick for backprop through sampling
- Tanh squashing to bound actions
- Log probability correction for squashing

### 2. Twin Q-Networks (Critics)

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # Q1
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2
```

**Why Twin Q-Networks?**

- Reduces overestimation bias (like TD3)
- Use minimum of two Q-values for target
- Improves stability and final performance

### 3. Automatic Temperature Tuning

```python
class SAC:
    def __init__(self, state_dim, action_dim, target_entropy=None):
        # ... networks ...

        # Automatic temperature tuning
        if target_entropy is None:
            self.target_entropy = -action_dim  # Heuristic
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], lr=3e-4)

    @property
    def alpha(self):
        return self.log_alpha.exp()
```

**Temperature Update:**

```python
def update_temperature(self, log_probs):
    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

    self.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.alpha_optimizer.step()
```

## 🔄 Training Algorithm

### Complete SAC Training Loop

```python
def train_sac(env, agent, num_steps=1e6, batch_size=256):
    replay_buffer = ReplayBuffer(capacity=1e6)
    state = env.reset()

    for step in range(int(num_steps)):
        # Select action
        if step < 10000:  # Random warmup
            action = env.action_space.sample()
        else:
            action, _, _ = agent.policy.sample(state)
            action = action.detach().cpu().numpy()

        # Environment step
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

        # Train
        if len(replay_buffer) > batch_size:
            # Sample batch
            batch = replay_buffer.sample(batch_size)

            # Update critics
            q_loss = agent.update_critic(batch)

            # Update policy
            policy_loss, alpha_loss = agent.update_policy(batch)

            # Update target networks
            agent.soft_update_targets()
```

### Critic Update

```python
def update_critic(self, batch):
    states, actions, rewards, next_states, dones = batch

    with torch.no_grad():
        # Sample actions from current policy
        next_actions, next_log_probs, _ = self.policy.sample(next_states)

        # Compute target Q-values (use minimum of two Q-networks)
        q1_next, q2_next = self.target_q(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next)

        # Add entropy term
        q_target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_probs)

    # Current Q-values
    q1, q2 = self.q_network(states, actions)

    # MSE loss
    q1_loss = F.mse_loss(q1, q_target)
    q2_loss = F.mse_loss(q2, q_target)
    q_loss = q1_loss + q2_loss

    # Update
    self.q_optimizer.zero_grad()
    q_loss.backward()
    self.q_optimizer.step()

    return q_loss.item()
```

### Policy Update

```python
def update_policy(self, batch):
    states = batch.states

    # Sample actions from current policy
    actions, log_probs, _ = self.policy.sample(states)

    # Compute Q-values
    q1, q2 = self.q_network(states, actions)
    q = torch.min(q1, q2)

    # Policy loss: maximize Q - α*entropy
    policy_loss = (self.alpha * log_probs - q).mean()

    # Update policy
    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    # Update temperature
    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
    self.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.alpha_optimizer.step()

    return policy_loss.item(), alpha_loss.item()
```

### Soft Target Update

```python
def soft_update_targets(self, tau=0.005):
    for target_param, param in zip(self.target_q.parameters(), self.q_network.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

## 🎮 Environments

### Recommended Environments

1. **Pendulum-v1**: Simple continuous control
2. **HalfCheetah-v3**: Locomotion task
3. **Ant-v3**: More complex locomotion
4. **Humanoid-v3**: Challenging (optional)

### Expected Performance

| Environment    | Random | SAC Target | Episodes  |
| -------------- | ------ | ---------- | --------- |
| Pendulum-v1    | -1500  | -200       | 100-200   |
| HalfCheetah-v3 | -200   | 10000+     | 1000-2000 |
| Ant-v3         | 0      | 5000+      | 2000-3000 |

## 🔬 Experiments

### Required Experiments

1. **Learning Curves**: Compare SAC, PPO, DDPG
2. **Ablation Studies**:
   - Without entropy (α=0)
   - Without twin Q-networks
   - Fixed vs automatic temperature
3. **Hyperparameter Sensitivity**:
   - Learning rates
   - Target entropy
   - Network sizes

### Analysis Questions

1. How does entropy regularization affect exploration?
2. Impact of target entropy on final performance?
3. Why do twin Q-networks help?
4. When does SAC outperform PPO/DDPG?

## 💡 Implementation Tips

- ✅ Use target entropy = -dim(action_space)
- ✅ Start with random policy warmup (10k steps)
- ✅ Normalize observations
- ✅ Use gradient clipping
- ✅ Monitor entropy during training
- ✅ Use large replay buffer (1M)

## 📖 References

### Key Papers

1. **Haarnoja et al. (2018)** - Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL

   - [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

2. **Haarnoja et al. (2018)** - SAC Algorithms and Applications

   - [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)

3. **Ziebart et al. (2008)** - Maximum Entropy Inverse RL
   - Foundation for MaxEnt RL

### Resources

- [Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [Stable Baselines3: SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)

## ⏱️ Time Estimate

- **Implementation**: 10-15 hours
- **Training & Experiments**: 8-12 hours
- **Analysis & Report**: 3-5 hours
- **Total**: 21-32 hours

---

**Difficulty**: ⭐⭐⭐⭐⭐ (Advanced)  
**Prerequisites**: HW1-3, Stochastic optimization  
**Key Skills**: MaxEnt RL, advanced actor-critic

SAC represents the state-of-the-art in continuous control. Enjoy implementing it!

