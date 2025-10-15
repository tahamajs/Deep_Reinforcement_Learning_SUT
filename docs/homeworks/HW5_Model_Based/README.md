# HW5: Model-Based Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Model-Based](https://img.shields.io/badge/Methods-Model--Based-purple.svg)](https://www.deepmind.com/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment explores **model-based reinforcement learning**, where agents learn a model of the environment and use it for planning. Unlike model-free methods that learn policies or value functions directly from experience, model-based methods learn environment dynamics and leverage them for more sample-efficient learning.

## 🎯 Learning Objectives

1. **Understand Environment Modeling**: Learn to predict state transitions and rewards
2. **Dyna Architecture**: Combine model-learning with model-free RL
3. **Monte Carlo Tree Search (MCTS)**: Master planning through simulation
4. **Model Predictive Control (MPC)**: Learn receding horizon optimization
5. **Sample Efficiency**: Compare model-based vs model-free sample complexity
6. **Planning vs Learning**: Understand trade-offs between planning and learning

## 📂 Directory Structure

```
HW5_Model_Based/
├── code/
│   ├── RL_HW5_Dyna.ipynb           # Dyna-Q algorithm
│   ├── RL_HW5_MCTS.ipynb            # Monte Carlo Tree Search
│   └── RL_HW5_MPC.ipynb             # Model Predictive Control
├── answers/
│   ├── RL_HW5_Dyna_Solution.ipynb
│   ├── RL_HW5_MPC_Solution.ipynb
│   └── HW5_Solution.pdf
├── reports/
│   └── HW5_Questions.pdf
└── README.md
```

## 📚 Theoretical Background

### 1. Model-Based RL Framework

**Core Components:**
```
Model: f(s,a) → (s', r)  or  P(s'|s,a), R(s,a,s')
Planner: Uses model to select actions
Policy: May be explicit or implicit through planning
```

**Advantages:**
- **Sample Efficiency**: Can learn from fewer environment interactions
- **Transfer**: Models can generalize across tasks
- **Safety**: Can simulate before acting
- **Interpretability**: Explicit dynamics understanding

**Challenges:**
- **Model Errors**: Compounding errors during planning
- **Computational Cost**: Planning can be expensive
- **Model Complexity**: Real environments are hard to model

### 2. Dyna Architecture

**Key Idea:** Integrate learning and planning by using both real and simulated experience.

**Components:**
1. **Direct RL**: Learn from real experience
2. **Model Learning**: Learn environment model
3. **Planning**: Generate simulated experience using model

**Dyna-Q Algorithm:**
```python
# Initialize
Q(s,a) = 0 for all s,a
Model(s,a) = None for all s,a

for episode in range(N):
    s = initial_state
    
    while not done:
        # 1. Direct RL: Act in real environment
        a = ε-greedy(Q, s)
        s', r = env.step(a)
        
        # 2. Update Q-function (Q-learning)
        Q(s,a) += α[r + γ max Q(s',a') - Q(s,a)]
                        a'
        
        # 3. Model Learning: Update model
        Model(s,a) = (s', r)
        
        # 4. Planning: n simulated updates
        for _ in range(n_planning_steps):
            # Sample previously visited state-action
            s_sim = random_visited_state()
            a_sim = random_action(s_sim)
            
            # Get predicted next state and reward
            s'_sim, r_sim = Model(s_sim, a_sim)
            
            # Q-learning update with simulated experience
            Q(s_sim, a_sim) += α[r_sim + γ max Q(s'_sim, a') - Q(s_sim, a_sim)]
                                          a'
        
        s = s'
```

**Why Dyna Works:**
- Real experience updates both Q and model
- Model generates "free" experience for Q-learning
- More updates per real interaction → faster learning

**Dyna Variants:**
- **Dyna-Q**: Q-learning as RL component
- **Dyna-Q+**: Adds exploration bonus for stale state-actions
- **Dyna-AC**: Actor-Critic as RL component

### 3. Monte Carlo Tree Search (MCTS)

**Key Idea:** Build search tree incrementally through simulation, focusing on promising regions.

**Four Phases per Simulation:**

#### 1. Selection
```
Traverse tree using UCB1:
a* = argmax [Q(s,a) + c√(ln N(s)/N(s,a))]
      a
      
where:
- Q(s,a) = average value (exploitation)
- c√(ln N(s)/N(s,a)) = exploration bonus
- N(s) = visit count of state s
- N(s,a) = visit count of action a in state s
```

#### 2. Expansion
Add new node to tree when leaf reached

#### 3. Simulation (Rollout)
```python
def rollout(state):
    total_reward = 0
    discount = 1.0
    
    while not is_terminal(state):
        action = random_policy(state)
        state, reward = model.step(state, action)
        total_reward += discount * reward
        discount *= gamma
    
    return total_reward
```

#### 4. Backpropagation
```python
# Update all nodes in path
for (state, action) in path:
    N(state, action) += 1
    Q(state, action) += (value - Q(state, action)) / N(state, action)
```

**MCTS Algorithm:**
```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

def mcts(root_state, n_simulations):
    root = MCTSNode(root_state)
    
    for _ in range(n_simulations):
        # 1. Selection
        node = root
        path = [node]
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child(c=√2)
            path.append(node)
        
        # 2. Expansion
        if not node.is_terminal():
            node = node.expand()
            path.append(node)
        
        # 3. Simulation
        value = rollout(node.state)
        
        # 4. Backpropagation
        for node in reversed(path):
            node.visits += 1
            node.value += (value - node.value) / node.visits
    
    # Return best action
    return max(root.children.items(), 
               key=lambda x: x[1].visits)[0]
```

**Applications:**
- **Go (AlphaGo)**: Combined with neural networks
- **Game Playing**: Chess, Poker, etc.
- **Robotics**: Motion planning
- **MDP Planning**: When model available

### 4. Model Predictive Control (MPC)

**Key Idea:** Repeatedly solve finite-horizon optimization, execute first action, replan.

**Algorithm:**
```
At time t:
1. Optimize action sequence [at, at+1, ..., at+H-1]
   to maximize predicted returns
2. Execute only first action at
3. Observe new state st+1
4. Repeat from step 1 (receding horizon)
```

**Mathematical Formulation:**
```
at* = argmax ∑k=0^H-1 γᵏ r(st+k, at+k)
      at,...,at+H-1

subject to: st+k+1 = f(st+k, at+k)  (model)
            constraints on actions/states
```

**Optimization Methods:**

#### a) Random Shooting
```python
def random_shooting_mpc(state, model, horizon, n_samples):
    best_value = -inf
    best_action = None
    
    for _ in range(n_samples):
        # Sample random action sequence
        actions = [random_action() for _ in range(horizon)]
        
        # Rollout using model
        value = 0
        s = state
        discount = 1.0
        
        for a in actions:
            s, r = model.step(s, a)
            value += discount * r
            discount *= gamma
        
        # Track best
        if value > best_value:
            best_value = value
            best_action = actions[0]
    
    return best_action
```

#### b) Cross-Entropy Method (CEM)
```python
def cem_mpc(state, model, horizon, n_samples, n_elite):
    # Initialize distribution
    mean = zeros(horizon * action_dim)
    std = ones(horizon * action_dim)
    
    for iteration in range(n_iterations):
        # Sample action sequences
        samples = normal(mean, std, size=(n_samples, horizon*action_dim))
        
        # Evaluate each sequence
        values = [evaluate(state, model, actions) 
                  for actions in samples]
        
        # Select elite samples
        elite_idxs = argsort(values)[-n_elite:]
        elite_samples = samples[elite_idxs]
        
        # Update distribution
        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)
    
    # Return first action of best sequence
    return mean[:action_dim]
```

**MPC Advantages:**
- Works with learned models (even imperfect ones)
- Handles constraints naturally
- Computationally parallelizable
- No policy network needed

**MPC Challenges:**
- Computationally expensive (must optimize at each step)
- Requires differentiable model (for gradient-based optimization)
- Horizon selection critical
- Compounding model errors

### 5. Comparison

| Method | Planning | Learning | Sample Efficiency | Computation |
|--------|----------|----------|-------------------|-------------|
| **Dyna** | Simulated updates | Yes | High | Low |
| **MCTS** | Tree search | Optional | High | Medium |
| **MPC** | Optimization | Model only | Very High | High |
| **Model-Free** | No | Yes | Low | Low |

## 💻 Implementation Details

### Part 1: Dyna-Q

**Environment:** GridWorld or MountainCar

**Tasks:**
1. Implement tabular Dyna-Q
2. Vary n (planning steps per real step)
3. Compare with vanilla Q-learning
4. Test on changing environments (Dyna-Q+ exploration bonus)

**Key Metrics:**
- Episodes to convergence vs n
- Sample efficiency improvement
- Performance on non-stationary environments

### Part 2: MCTS

**Environment:** Games (TicTacToe, ConnectFour) or planning tasks

**Tasks:**
1. Implement MCTS with UCB1 selection
2. Tune exploration constant c
3. Compare with minimax or random play
4. Analyze search tree growth

**Visualization:**
- Tree structure after N simulations
- Visit counts vs value estimates
- Action selection over time

### Part 3: MPC

**Environment:** Continuous control (Pendulum, CartPole)

**Tasks:**
1. Learn dynamics model (neural network)
2. Implement random shooting MPC
3. Implement CEM-based MPC
4. Compare with model-free baseline

**Model Architecture:**
```python
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_dim + 1)  # next_state + reward
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        
        next_state_delta = output[:, :-1]
        reward = output[:, -1]
        
        next_state = state + next_state_delta  # Predict change
        return next_state, reward
```

## 📊 Evaluation Metrics

1. **Sample Efficiency**: Performance vs real environment steps
2. **Planning Time**: Computation per action selection
3. **Model Accuracy**: Prediction error on test transitions
4. **Asymptotic Performance**: Final policy quality
5. **Robustness**: Performance as model error increases

## 🔧 Requirements

```python
numpy>=1.21.0
matplotlib>=3.4.0
torch>=2.0.0
gymnasium>=0.28.0
scipy>=1.7.0
```

## 🚀 Getting Started

```bash
cd HW5_Model_Based
pip install -r requirements.txt

jupyter notebook code/RL_HW5_Dyna.ipynb
jupyter notebook code/RL_HW5_MCTS.ipynb
jupyter notebook code/RL_HW5_MPC.ipynb
```

## 📈 Expected Results

### Dyna-Q
- 5-10x fewer episodes than Q-learning
- Improvement scales with planning steps
- Better performance in sparse reward environments

### MCTS
- Superhuman performance in games
- Tree pruning improves efficiency
- Performance improves with simulation budget

### MPC
- Can solve tasks with ~100 samples for model learning
- CEM outperforms random shooting
- Longer horizons → better but slower

## 📖 References

1. **Sutton, R. S. (1990)** - "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming" - *ICML*

2. **Browne, C. B., et al. (2012)** - "A survey of monte carlo tree search methods" - *IEEE TCIAIG*

3. **Silver, D., et al. (2016)** - "Mastering the game of Go with deep neural networks and tree search" - *Nature*

4. **Nagabandi, A., et al. (2018)** - "Neural Network Dynamics for Model-Based Deep RL" - *ICML*

## 💡 Discussion Questions

1. When does model-based RL outperform model-free?
2. How do model errors compound during planning?
3. Why does MCTS work well for games but struggle in continuous domains?
4. What are the trade-offs between planning horizon and computational cost?

## 🎓 Extensions

- Implement World Models (Ha & Schmidhuber, 2018)
- Try ensemble models for uncertainty estimation
- Implement MBPO (Model-Based Policy Optimization)
- Add model-based exploration bonuses

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024

