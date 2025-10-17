# HW2: Policy Gradient Methods - PPO vs DDPG

## 📋 Overview

This assignment introduces policy gradient methods, focusing on two state-of-the-art algorithms: Proximal Policy Optimization (PPO) for discrete actions and Deep Deterministic Policy Gradient (DDPG) for continuous control.

## 📂 Contents

```
HW2/
├── SP23_RL_HW2/
│   ├── RL_HW2.pdf                    # Assignment description
│   └── RL_HW2_PPO_vs_DDPG.ipynb     # Implementation notebook
├── RL_HW2_Solution.pdf               # Complete solutions
└── README.md                          # This file
```

## 🎯 Learning Objectives

- ✅ Understand policy gradient theorem
- ✅ Implement PPO with clipped objective
- ✅ Implement DDPG for continuous control
- ✅ Master advantage estimation (GAE)
- ✅ Compare on-policy vs off-policy methods
- ✅ Analyze convergence and stability

## 📚 Background

### Why Policy Gradients?

**Value-based methods (DQN):**

- Learn Q(s,a) → derive policy
- Works well for discrete actions
- Struggles with continuous action spaces

**Policy-based methods:**

- Learn π(a|s) directly
- Natural for continuous actions
- Can learn stochastic policies
- Better convergence properties

### Policy Gradient Theorem

**Objective:** Maximize expected return

```
J(θ) = E_{τ~π_θ}[∑_t r_t]
```

**Gradient:**

```
∇_θ J(θ) = E_{τ~π_θ}[∑_t ∇_θ log π_θ(a_t|s_t) G_t]
```

**With Baseline:**

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) A^π(s,a)]
```

where A^π(s,a) = Q^π(s,a) - V^π(s) is the advantage function.

## 🚀 Part 1: Proximal Policy Optimization (PPO)

### Algorithm Overview

PPO is an on-policy algorithm that:

- Uses importance sampling for multiple update epochs
- Clips policy updates to avoid destructive large changes
- Combines policy gradient with value function learning

### Key Components

#### 1.1 Policy Network (Actor)

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
```

#### 1.2 Value Network (Critic)

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)
```

#### 1.3 Generalized Advantage Estimation (GAE)

```python
def compute_gae(rewards, values, next_value, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
        next_value = values[t]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
```

**Why GAE?**

- Reduces variance while controlling bias
- λ=0: low variance, high bias (TD)
- λ=1: high variance, low bias (MC)
- λ=0.95: good balance

#### 1.4 PPO Clipped Objective

```python
def ppo_loss(policy, old_policy, states, actions, advantages, epsilon=0.2):
    # Current policy
    dist = policy(states)
    log_probs = dist.log_prob(actions)

    # Old policy (fixed)
    old_dist = old_policy(states)
    old_log_probs = old_dist.log_prob(actions).detach()

    # Importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped objective
    clip_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(
        ratio * advantages,
        clip_ratio * advantages
    ).mean()

    return policy_loss
```

**Why Clipping?**

- Prevents too large policy updates
- Maintains trust region
- Improves training stability
- Simple alternative to TRPO

#### 1.5 Complete PPO Algorithm

```python
# Hyperparameters
gamma = 0.99
lambda_gae = 0.95
epsilon_clip = 0.2
ppo_epochs = 10
batch_size = 64

for iteration in range(num_iterations):
    # Collect trajectories
    states, actions, rewards, values = collect_trajectories(policy, env)

    # Compute advantages and returns
    advantages, returns = compute_gae(rewards, values, gamma, lambda_gae)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Multiple epochs of optimization
    for epoch in range(ppo_epochs):
        # Sample mini-batches
        for batch in get_batches(states, actions, advantages, returns):
            # Policy loss
            policy_loss = ppo_loss(policy, old_policy, batch, epsilon_clip)

            # Value loss
            value_pred = value_net(batch.states)
            value_loss = F.mse_loss(value_pred, batch.returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update old policy
    old_policy.load_state_dict(policy.state_dict())
```

## 🎮 Part 2: Deep Deterministic Policy Gradient (DDPG)

### Algorithm Overview

DDPG is an off-policy actor-critic algorithm for continuous control:

- Deterministic policy gradient
- Experience replay
- Target networks for stability
- Exploration via noise

### Key Components

#### 2.1 Actor Network (Deterministic Policy)

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action
```

#### 2.2 Critic Network (Q-function)

```python
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
```

#### 2.3 Ornstein-Uhlenbeck Noise

```python
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
```

**Why OU Noise?**

- Temporally correlated noise
- Better exploration than Gaussian
- Mean-reverting process

#### 2.4 DDPG Training Loop

```python
# Initialize networks
actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
actor_target = copy.deepcopy(actor)
critic_target = copy.deepcopy(critic)

# Optimizers
actor_optimizer = Adam(actor.parameters(), lr=1e-4)
critic_optimizer = Adam(critic.parameters(), lr=1e-3)

# Replay buffer and noise
replay_buffer = ReplayBuffer(capacity=1e6)
noise = OUNoise(action_dim)

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Select action with exploration noise
        action = actor(state).detach() + noise.sample()
        action = np.clip(action, -max_action, max_action)

        # Environment step
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        # Train
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)

            # Critic update
            with torch.no_grad():
                next_action = actor_target(batch.next_states)
                target_q = critic_target(batch.next_states, next_action)
                target_q = batch.rewards + gamma * target_q * (1 - batch.dones)

            current_q = critic(batch.states, batch.actions)
            critic_loss = F.mse_loss(current_q, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor update
            actor_loss = -critic(batch.states, actor(batch.states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update target networks
            soft_update(actor_target, actor, tau=0.005)
            soft_update(critic_target, critic, tau=0.005)
```

#### 2.5 Soft Update

```python
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data
        )
```

## 📊 Environments

### For PPO (Discrete Actions)

- **CartPole-v1**: Simple balancing task
- **LunarLander-v2**: Spacecraft landing
- **Pong**: Atari game (optional)

### For DDPG (Continuous Actions)

- **Pendulum-v1**: Swing up and balance
- **MountainCarContinuous-v0**: Drive up hill
- **BipedalWalker-v3**: Walking robot (challenging)

## 🔬 Experiments and Analysis

### Required Experiments

#### 1. Learning Curves

- Plot episode reward vs episode
- Compare PPO vs DDPG on same environment
- Show mean ± std over 5 seeds

#### 2. Hyperparameter Sensitivity

**PPO:**

- Clip epsilon: [0.1, 0.2, 0.3]
- GAE lambda: [0.9, 0.95, 0.99]
- PPO epochs: [5, 10, 20]

**DDPG:**

- Tau (soft update): [0.001, 0.005, 0.01]
- Noise sigma: [0.1, 0.2, 0.3]
- Actor vs critic learning rates

#### 3. Ablation Studies

**PPO without clipping:**

```python
# Remove clipping to see importance
policy_loss = -(ratio * advantages).mean()
```

**DDPG without target networks:**

```python
# Use same network for targets
target_q = critic(next_states, actor(next_states))
```

### Analysis Questions

1. **Convergence Speed**: Which algorithm converges faster?
2. **Sample Efficiency**: Episodes needed to solve
3. **Stability**: Variance across random seeds
4. **Final Performance**: Maximum achieved reward
5. **Computational Cost**: Training time comparison

## 💡 Implementation Tips

### PPO Tips

- ✅ Normalize advantages (zero mean, unit variance)
- ✅ Use orthogonal initialization for networks
- ✅ Clip value function loss
- ✅ Add entropy bonus for exploration
- ✅ Use learning rate annealing

### DDPG Tips

- ✅ Normalize state inputs
- ✅ Use batch normalization
- ✅ Start with small noise, decay over time
- ✅ Use different learning rates for actor/critic
- ✅ Warm up replay buffer before training

### Debugging

**PPO not learning:**

- Check advantage computation
- Verify clip range not too small
- Ensure sufficient exploration
- Monitor KL divergence

**DDPG unstable:**

- Reduce learning rates
- Increase target network tau
- Add gradient clipping
- Check action bounds

## 📖 References

### Key Papers

1. **Schulman et al. (2017)** - Proximal Policy Optimization

   - [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

2. **Lillicrap et al. (2015)** - Continuous Control with Deep RL

   - [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

3. **Schulman et al. (2015)** - High-Dimensional Continuous Control Using GAE
   - [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

### Additional Reading

- Sutton & Barto Chapter 13 (Policy Gradient Methods)
- OpenAI Spinning Up: [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

## ⏱️ Time Estimate

- **Part 1 (PPO)**: 8-10 hours
- **Part 2 (DDPG)**: 8-10 hours
- **Experiments**: 4-6 hours
- **Analysis**: 2-3 hours
- **Total**: 22-29 hours

## 📧 Getting Help

- **Office Hours**: Algorithm details and debugging
- **Piazza**: Conceptual questions
- **Study Groups**: Compare implementations

---

**Difficulty**: ⭐⭐⭐⭐☆ (Challenging)  
**Prerequisites**: HW1, Policy gradients lecture  
**Key Skills**: Advanced RL algorithms, continuous control

This is a challenging but rewarding assignment. Start early!

