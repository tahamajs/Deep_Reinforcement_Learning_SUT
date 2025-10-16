# HW2: Value-Based Deep Reinforcement Learning

## 📋 Overview

This assignment bridges tabular methods and deep RL, implementing Monte Carlo, TD methods, and Deep Q-Networks with experience replay and target networks.

## 📂 Contents

```
HW2/
├── SP24_RL_HW2/
│   ├── RL_HW2.pdf            # Assignment description
│   ├── RL_HW2_MC_TD.ipynb    # Part 1: Monte Carlo & TD
│   └── RL_HW2_DQN.ipynb      # Part 2: Deep Q-Networks
├── RL_HW2_Solution.pdf        # Solutions
└── README.md                   # This file
```

## 🎯 Learning Objectives

### Part 1: MC and TD Methods

- ✅ Implement first-visit Monte Carlo
- ✅ Implement TD(0) and TD(λ)
- ✅ Compare MC vs TD trade-offs
- ✅ Understand eligibility traces

### Part 2: Deep Q-Networks

- ✅ Build neural network Q-functions
- ✅ Implement experience replay
- ✅ Use target networks
- ✅ Train DQN and Double DQN
- ✅ Analyze overestimation bias

## 📚 Part 1: Monte Carlo and TD Methods

### Monte Carlo

**First-Visit MC:**

```python
def first_visit_mc(env, policy, num_episodes, gamma=0.99):
    V = defaultdict(float)
    returns = defaultdict(list)

    for episode in range(num_episodes):
        states, rewards = generate_episode(env, policy)
        G = 0
        visited = set()

        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            s = states[t]

            if s not in visited:
                visited.add(s)
                returns[s].append(G)
                V[s] = np.mean(returns[s])

    return V
```

### Temporal Difference

**TD(0):**

```python
def td_zero(env, policy, num_episodes, alpha=0.1, gamma=0.99):
    V = defaultdict(float)

    for episode in range(num_episodes):
        state = env.reset()

        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)

            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state

    return V
```

**TD(λ) with Eligibility Traces:**

```python
def td_lambda(env, policy, num_episodes, alpha=0.1, gamma=0.99, lambda_=0.9):
    V = defaultdict(float)

    for episode in range(num_episodes):
        eligibility = defaultdict(float)
        state = env.reset()

        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1

            for s in eligibility:
                V[s] += alpha * delta * eligibility[s]
                eligibility[s] *= gamma * lambda_

            state = next_state

    return V
```

### Comparison

**MC vs TD:**
| Aspect | Monte Carlo | TD Learning |
|--------|-------------|-------------|
| Bootstrapping | No | Yes |
| Bias | Unbiased | Biased |
| Variance | High | Low |
| Online | No | Yes |
| Convergence | Slower | Faster |

## 📚 Part 2: Deep Q-Networks

### DQN Architecture

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, state):
        return self.network(state)
```

### Experience Replay

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

### DQN Training

```python
def train_dqn(env, policy_net, target_net, num_episodes=500):
    replay_buffer = ReplayBuffer()
    optimizer = Adam(policy_net.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # ε-greedy action selection
            action = select_action(state, policy_net, epsilon)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            # Train
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_dqn_loss(policy_net, target_net, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            state = next_state
            if done:
                break
```

### Double DQN

```python
def compute_ddqn_loss(policy_net, target_net, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # Current Q-values
    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        # Use policy net to SELECT action
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        # Use target net to EVALUATE action
        next_q_values = target_net(next_states).gather(1, next_actions)
        targets = rewards + gamma * next_q_values * (1 - dones)

    return F.mse_loss(q_values, targets)
```

## 🔬 Experiments

### Required Experiments

1. **MC vs TD Comparison**

   - Convergence speed
   - Final accuracy
   - Variance analysis

2. **DQN vs Double DQN**

   - Learning curves
   - Q-value estimates
   - Overestimation analysis

3. **Hyperparameter Sensitivity**
   - Learning rate: [1e-4, 1e-3, 1e-2]
   - Batch size: [32, 64, 128]
   - Target update frequency

## 📊 Expected Results

**CartPole-v1:**

- Random agent: ~20 steps
- DQN (trained): 400-500 steps
- Solved: 195+ average over 100 episodes

**CliffWalking:**

- Optimal reward: -13
- Q-Learning: -13 to -15
- SARSA: -15 to -25 (safer)

## 💡 Implementation Tips

- Start with small buffer (10k), increase if needed
- Use ε-decay: 1.0 → 0.01 over 1000 episodes
- Update target network every 100-1000 steps
- Monitor Q-values for overestimation
- Plot moving average (window=100)

## 📖 References

- **Mnih et al. (2015)** - DQN Nature Paper
- **van Hasselt et al. (2016)** - Double DQN
- **Sutton & Barto (2018)** - Chapters 5-6, 9

## ⏱️ Time Estimate

- Part 1 (MC/TD): 6-8 hours
- Part 2 (DQN): 10-14 hours
- Analysis: 3-5 hours
- **Total**: 19-27 hours

---

**Difficulty**: ⭐⭐⭐☆☆ (Moderate)  
**Prerequisites**: HW1, PyTorch basics  
**Key Skills**: Function approximation, deep RL

