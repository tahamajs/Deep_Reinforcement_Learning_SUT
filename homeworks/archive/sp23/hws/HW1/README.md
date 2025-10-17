# HW1: Tabular Methods and Deep Q-Learning

## 📋 Overview

This assignment bridges classical tabular RL methods and modern deep reinforcement learning. You'll implement foundational algorithms from scratch and apply them to both tabular and continuous state spaces.

## 📂 Contents

```
HW1/
├── SP23_RL_HW1/
│   ├── RL_HW1.pdf                    # Assignment description
│   ├── RL_HW1_Tabular.ipynb          # Part 1: Tabular methods
│   └── RL_HW1_CartPole.ipynb         # Part 2: DQN
├── SP23_RL_HW1_Solutions/
│   ├── RL_HW1_Solution.pdf           # Written solutions
│   ├── RL_HW1_Tabular_Solutions.ipynb
│   └── RL_HW1_CartPole_Solutions.ipynb
└── README.md                          # This file
```

## 🎯 Learning Objectives

### Part 1: Tabular Methods

- ✅ Implement Value Iteration from scratch
- ✅ Implement Policy Iteration from scratch
- ✅ Implement Q-Learning algorithm
- ✅ Understand convergence properties
- ✅ Compare algorithm efficiency

### Part 2: Deep Q-Learning

- ✅ Build neural network Q-function approximators
- ✅ Implement experience replay mechanism
- ✅ Use target networks for stability
- ✅ Train agents on CartPole environment
- ✅ Analyze learning curves and performance

## 📚 Part 1: Tabular Methods

### Environment: GridWorld

**Description:**

- N×M grid of states
- Agent can move: up, down, left, right
- Goal state with positive reward
- Obstacle states with negative reward
- Episode ends at goal or obstacle

### 1.1 Value Iteration

**Algorithm:**

```python
# Initialize
V = zeros(n_states)

# Iterate
while not converged:
    for each state s:
        V_new[s] = max_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV[s']]
    V = V_new

# Extract policy
π[s] = argmax_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV[s']]
```

**Key Features:**

- Finds optimal value function V\*
- Uses Bellman optimality operator
- Converges to optimal policy
- Requires model (P and R)

**Implementation Tasks:**

1. Initialize value function
2. Implement Bellman optimality update
3. Check convergence (max change < θ)
4. Extract greedy policy from V\*

**Expected Results:**

- Convergence in 10-50 iterations
- Optimal policy reaches goal efficiently
- Higher γ → more iterations needed

### 1.2 Policy Iteration

**Algorithm:**

```python
# Initialize
π = random_policy()

while not converged:
    # Policy Evaluation
    V_π = evaluate_policy(π)

    # Policy Improvement
    π_new[s] = argmax_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV_π[s']]

    if π_new == π:
        break
    π = π_new
```

**Key Features:**

- Alternates evaluation and improvement
- Guaranteed to converge to π\*
- Often fewer iterations than VI
- Each iteration more expensive

**Implementation Tasks:**

1. Policy evaluation (solve system or iterate)
2. Policy improvement step
3. Check if policy changed
4. Compare with value iteration

**Expected Results:**

- Converges in 3-10 policy iterations
- Each iteration requires solving V^π
- Same final policy as value iteration

### 1.3 Q-Learning

**Algorithm:**

```python
# Initialize
Q = zeros(n_states, n_actions)

for each episode:
    s = initial_state
    while not terminal:
        a = epsilon_greedy(s, Q)
        s', r = env.step(a)
        Q[s,a] += α[r + γ max_a' Q[s',a'] - Q[s,a]]
        s = s'
```

**Key Features:**

- Model-free (no P or R needed)
- Off-policy (learns π\* while following ε-greedy)
- Stochastic approximation
- Requires exploration

**Implementation Tasks:**

1. Initialize Q-table
2. Implement ε-greedy policy
3. Q-value update rule
4. Learning rate schedule
5. Exploration decay

**Expected Results:**

- Converges to Q\* with sufficient exploration
- Requires many more episodes than DP
- Sensitive to learning rate α
- Performance improves as ε decays

## 📚 Part 2: Deep Q-Network (DQN)

### Environment: CartPole-v1

**Description:**

- Continuous 4D state space
- 2 discrete actions (left/right)
- Goal: Balance pole for 500 timesteps
- Episode ends if pole falls or cart exits bounds

**State:** [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

### 2.1 Q-Network Architecture

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action
```

**Architecture Choices:**

- Input: State vector (4D)
- Hidden: 128 → 128 neurons
- Output: Q-values (2D)
- Activation: ReLU
- No activation on output layer

### 2.2 Experience Replay

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        return random.sample(self.buffer, batch_size)
```

**Why Replay?**

- Breaks temporal correlations
- Improves sample efficiency
- Enables mini-batch SGD
- Stabilizes training

**Hyperparameters:**

- Buffer size: 10,000 - 100,000
- Batch size: 32 - 64
- Start learning after: 1,000 samples

### 2.3 Target Network

```python
# Create policy and target networks
policy_net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(policy_net.state_dict())

# Training loop
for step in steps:
    # ... collect experience ...

    # Update policy network
    loss = compute_loss(policy_net, target_net, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Periodically update target network
    if step % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
```

**Why Target Network?**

- Provides stable TD targets
- Prevents moving target problem
- Reduces oscillations
- Update every 100-1000 steps

### 2.4 DQN Loss Function

```python
def compute_dqn_loss(policy_net, target_net, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # Current Q-values
    q_values = policy_net(states).gather(1, actions)

    # Target Q-values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

    # MSE loss
    loss = F.mse_loss(q_values, targets)
    return loss
```

### 2.5 Complete DQN Algorithm

```python
# Initialize
policy_net = QNetwork()
target_net = QNetwork()
replay_buffer = ReplayBuffer(10000)
optimizer = Adam(policy_net.parameters(), lr=1e-3)
epsilon = 1.0

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        # Environment step
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        # Train
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_dqn_loss(policy_net, target_net, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        episode_reward += reward

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
```

## 🔧 Implementation Tasks

### Part 1: Tabular Methods (30 points)

1. **Value Iteration** (10 pts)

   - Implement algorithm
   - Test on GridWorld
   - Plot value function heatmap

2. **Policy Iteration** (10 pts)

   - Policy evaluation step
   - Policy improvement step
   - Compare with value iteration

3. **Q-Learning** (10 pts)
   - Implement with ε-greedy
   - Train on GridWorld
   - Plot learning curves

### Part 2: DQN (40 points)

1. **Q-Network** (10 pts)

   - Build PyTorch network
   - Test forward pass

2. **Replay Buffer** (5 pts)

   - Implement data structure
   - Test push/sample

3. **DQN Agent** (15 pts)

   - Complete training loop
   - Implement target network
   - Train on CartPole

4. **Analysis** (10 pts)
   - Plot learning curves
   - Compare hyperparameters
   - Discuss results

### Written Questions (30 points)

- Convergence proofs
- Algorithm comparisons
- Hyperparameter analysis

## 📊 Expected Results

### GridWorld (Tabular)

- **Value Iteration**: 20-40 iterations to converge
- **Policy Iteration**: 5-15 iterations to converge
- **Q-Learning**: 1000-5000 episodes to converge

### CartPole (DQN)

- **Early training** (eps 0-100): 20-50 average reward
- **Mid training** (eps 100-300): 100-300 average reward
- **Late training** (eps 300-500): 400-500 average reward
- **Solved**: 195+ average reward over 100 episodes

## 🔬 Experiments

### Required Experiments

1. **Convergence Comparison**

   - Plot iterations to convergence
   - Compare VI vs PI vs Q-Learning
   - Vary discount factor γ

2. **Hyperparameter Sensitivity**

   - Learning rate α: [0.01, 0.1, 0.5]
   - Epsilon decay: [0.95, 0.99, 0.995]
   - Network size: [64, 128, 256]

3. **Learning Curves**
   - Episode reward vs episode
   - Moving average (window=100)
   - Multiple random seeds (3-5)

## 💡 Tips for Success

### Debugging Tabular Methods

- **Print intermediate values**: Check V and Q updates
- **Visualize policy**: Plot arrows showing action
- **Check convergence**: Monitor max value change
- **Verify**: Compare with analytical solutions on tiny grids

### Debugging DQN

- **Start simple**: Verify on CartPole first
- **Monitor metrics**: Loss, Q-values, reward
- **Check gradients**: Use `torch.nn.utils.clip_grad_norm_`
- **Visualize**: Plot Q-value evolution
- **Test replay buffer**: Ensure proper sampling

### Common Issues

**Issue 1: Q-Learning not converging**

- ✅ Increase episodes
- ✅ Tune learning rate
- ✅ Ensure sufficient exploration
- ✅ Check state representation

**Issue 2: DQN training unstable**

- ✅ Add gradient clipping
- ✅ Increase replay buffer size
- ✅ Update target network less frequently
- ✅ Normalize state inputs

**Issue 3: CartPole not solving**

- ✅ Train longer (500+ episodes)
- ✅ Tune epsilon decay
- ✅ Adjust learning rate
- ✅ Try different network sizes

## 📖 References

### Key Papers

1. **Watkins & Dayan (1992)** - Q-Learning
2. **Mnih et al. (2015)** - DQN (Nature paper)
3. **Sutton & Barto (2018)** - Chapters 4-6

### Code References

- [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [OpenAI Spinning Up - DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)

## ⏱️ Time Estimate

- **Part 1**: 6-8 hours
- **Part 2**: 8-12 hours
- **Written**: 2-3 hours
- **Total**: 16-23 hours

## 📧 Getting Help

- **Office Hours**: Debugging DQN training
- **Piazza**: Conceptual questions
- **Study Groups**: Compare implementations
- **TAs**: Code review and optimization

---

**Difficulty**: ⭐⭐⭐☆☆ (Moderate)  
**Prerequisites**: HW0, Python, PyTorch basics  
**Key Skills**: Dynamic programming, neural networks, debugging

Good luck! This assignment builds critical RL implementation skills.

