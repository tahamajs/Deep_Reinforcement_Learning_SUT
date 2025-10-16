# HW3: Model-Based RL and Multi-Armed Bandits

## 📋 Overview

This assignment explores model-based reinforcement learning through Monte Carlo Tree Search (MCTS) and multi-armed bandit algorithms, specifically Thompson Sampling. You'll implement planning algorithms and Bayesian approaches to exploration-exploitation.

## 📂 Contents

```
HW3/
├── SP23_RL_HW3/
│   ├── RL_HW3.pdf                      # Assignment description
│   ├── RL_HW3_MCTS.ipynb              # Part 1: Monte Carlo Tree Search
│   └── RL_HW3_Thompson_Sampling.ipynb # Part 2: Thompson Sampling
├── RL_HW3_solution.pdf                 # Complete solutions
└── README.md                            # This file
```

## 🎯 Learning Objectives

### Part 1: Monte Carlo Tree Search

- ✅ Understand tree-based planning
- ✅ Implement UCB1 for tree policy
- ✅ Apply MCTS to game playing
- ✅ Balance exploration and exploitation
- ✅ Analyze computational complexity

### Part 2: Thompson Sampling

- ✅ Master Bayesian bandits
- ✅ Implement Thompson Sampling
- ✅ Compare with UCB and ε-greedy
- ✅ Understand regret bounds
- ✅ Apply to real-world problems

---

## 🌲 Part 1: Monte Carlo Tree Search (MCTS)

### Background

MCTS is a best-first search algorithm that builds a search tree incrementally through random sampling. It's model-based and particularly effective for large state spaces.

**Key Applications:**

- Game playing (Go, Chess, Poker)
- Planning under uncertainty
- Combinatorial optimization

### Algorithm Components

MCTS consists of four phases repeated iteratively:

#### 1. Selection

```python
def select(node):
    """Select child using UCB1 until leaf node"""
    while not node.is_terminal() and node.is_fully_expanded():
        node = node.best_child(c_param=1.41)
    return node
```

#### 2. Expansion

```python
def expand(node):
    """Add a new child node"""
    untried_actions = node.untried_actions()
    action = random.choice(untried_actions)
    next_state = node.state.next_state(action)
    child = Node(state=next_state, parent=node, action=action)
    node.children.append(child)
    return child
```

#### 3. Simulation (Rollout)

```python
def simulate(node):
    """Random rollout until terminal state"""
    state = node.state
    while not state.is_terminal():
        action = state.random_action()
        state = state.next_state(action)
    return state.reward()
```

#### 4. Backpropagation

```python
def backpropagate(node, reward):
    """Update statistics up the tree"""
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent
```

### UCB1 (Upper Confidence Bound)

```python
def ucb1(node, c_param=1.41):
    """UCB1 formula for node selection"""
    if node.visits == 0:
        return float('inf')

    exploitation = node.reward / node.visits
    exploration = c_param * sqrt(log(node.parent.visits) / node.visits)
    return exploitation + exploration
```

**Formula:**

```
UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
```

- Q(s,a): Average reward from state-action
- N(s): Parent visit count
- N(s,a): Child visit count
- c: Exploration constant (typically √2)

### Complete MCTS Algorithm

```python
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def best_child(self, c_param=1.41):
        choices_weights = [
            (child.reward / child.visits) +
            c_param * sqrt(2 * log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def best_action(self):
        return max(self.children, key=lambda c: c.visits).action

def mcts(root_state, num_simulations=1000):
    root = MCTSNode(state=root_state)

    for _ in range(num_simulations):
        node = root

        # Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child()

        # Expansion
        if not node.is_terminal():
            node = node.expand()

        # Simulation
        reward = node.simulate()

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    return root.best_action()
```

### Environments

**Recommended Environments:**

1. **Tic-Tac-Toe**: Simple 3x3 game
2. **Connect Four**: More complex
3. **CartPole**: Continuous state MDP

### Analysis Questions

1. How does number of simulations affect performance?
2. Impact of exploration constant c?
3. Comparison with minimax/alpha-beta pruning?
4. When does MCTS outperform other planning methods?

---

## 🎰 Part 2: Thompson Sampling for Multi-Armed Bandits

### Background

Multi-armed bandits model the exploration-exploitation dilemma:

- K arms (actions), each with unknown reward distribution
- Goal: Maximize cumulative reward over T rounds
- Trade-off: Explore to learn vs exploit best known arm

### Problem Formulation

**Bernoulli Bandits:**

- Each arm i has success probability θᵢ
- Pull arm, observe reward r ∈ {0, 1}
- True θᵢ unknown, must be learned

**Regret:**

```
Regret(T) = T * μ* - ∑ᵗ₌₁ᵀ μ(aₜ)
```

where μ\* is the best arm's mean reward.

### Thompson Sampling Algorithm

#### Bayesian Framework

**Prior:** Beta(α, β) distribution for each arm

```
p(θᵢ) = Beta(αᵢ, βᵢ)
```

**Likelihood:** Bernoulli(θᵢ) for observed rewards

```
p(r|θᵢ) = θᵢʳ(1-θᵢ)¹⁻ʳ
```

**Posterior:** Beta(α + successes, β + failures)

```
p(θᵢ|data) = Beta(αᵢ + Sᵢ, βᵢ + Fᵢ)
```

#### Implementation

```python
class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Initialize with uniform prior Beta(1,1)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        """Sample from posterior and select best"""
        samples = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_arms)
        ]
        return np.argmax(samples)

    def update(self, arm, reward):
        """Update posterior based on observed reward"""
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

### Complete Bandit Experiment

```python
def run_bandit_experiment(algorithm, true_probs, n_rounds=1000):
    """
    Args:
        algorithm: Bandit algorithm (Thompson, UCB, epsilon-greedy)
        true_probs: True success probability for each arm
        n_rounds: Number of rounds to play
    """
    n_arms = len(true_probs)
    rewards = []
    regrets = []
    best_arm_mean = max(true_probs)

    for t in range(n_rounds):
        # Select arm
        arm = algorithm.select_arm()

        # Observe reward
        reward = np.random.binomial(1, true_probs[arm])

        # Update algorithm
        algorithm.update(arm, reward)

        # Track metrics
        rewards.append(reward)
        instantaneous_regret = best_arm_mean - true_probs[arm]
        regrets.append(instantaneous_regret)

    return {
        'rewards': rewards,
        'cumulative_regret': np.cumsum(regrets),
        'arm_counts': algorithm.get_arm_counts()
    }
```

### Comparison Algorithms

#### ε-Greedy

```python
class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
```

#### UCB1

```python
class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1

        # Try each arm once first
        if self.t <= self.n_arms:
            return self.t - 1

        # UCB1 formula
        ucb_values = [
            self.values[i] + sqrt(2 * log(self.t) / self.counts[i])
            for i in range(self.n_arms)
        ]
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
```

### Experiments

#### 1. Algorithm Comparison

Compare Thompson Sampling, UCB1, and ε-greedy on:

```python
# Easy problem: One good arm
true_probs_easy = [0.1, 0.1, 0.9, 0.1]

# Hard problem: Similar arms
true_probs_hard = [0.45, 0.48, 0.50, 0.47]
```

#### 2. Hyperparameter Sensitivity

- ε-greedy: ε ∈ {0.01, 0.05, 0.1, 0.2}
- Prior strength for Thompson Sampling

#### 3. Contextual Bandits (Optional)

Extend to contextual bandits where each arm's reward depends on context.

### Theoretical Results

**Thompson Sampling Regret:**

```
E[Regret(T)] = O(√(KT log T))
```

**UCB1 Regret:**

```
E[Regret(T)] = O(√(KT log T))
```

Both achieve logarithmic regret, which is optimal.

## 📊 Expected Results

### MCTS

- Tic-Tac-Toe: Near-optimal play with 1000+ simulations
- Win rate: >90% against random opponent
- Computation time: <1s per move

### Thompson Sampling

- Cumulative regret grows sublinearly
- Converges to best arm faster than ε-greedy
- Similar performance to UCB1
- Better in practice due to randomization

## 💡 Implementation Tips

### MCTS

- Start with small games (Tic-Tac-Toe)
- Vectorize rollouts for speed
- Tune exploration constant c
- Cache game states to avoid recomputation

### Thompson Sampling

- Use `np.random.beta` for sampling
- Verify posterior updates are correct
- Plot posterior distributions over time
- Test on synthetic problems first

## 📖 References

### Key Papers

1. **Browne et al. (2012)** - A Survey of Monte Carlo Tree Search Methods
2. **Kocsis & Szepesvári (2006)** - Bandit Based Monte-Carlo Planning
3. **Thompson (1933)** - On the likelihood that one unknown probability exceeds another
4. **Agrawal & Goyal (2012)** - Analysis of Thompson Sampling
5. **Auer et al. (2002)** - Finite-time Analysis of the Multiarmed Bandit Problem

### Additional Resources

- [MCTS Research Hub](http://mcts.ai/)
- [Bandit Algorithms Book](https://tor-lattimore.com/downloads/book/book.pdf)

## ⏱️ Time Estimate

- **Part 1 (MCTS)**: 8-12 hours
- **Part 2 (Thompson Sampling)**: 6-10 hours
- **Experiments & Analysis**: 4-6 hours
- **Total**: 18-28 hours

---

**Difficulty**: ⭐⭐⭐⭐☆ (Challenging)  
**Prerequisites**: HW1-2, Probability, Bayesian inference  
**Key Skills**: Planning, Bayesian methods, exploration-exploitation

Good luck exploring model-based RL and bandits!

