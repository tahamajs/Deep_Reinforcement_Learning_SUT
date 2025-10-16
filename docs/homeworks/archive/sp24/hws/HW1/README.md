# HW1: Introduction to Reinforcement Learning

## 📋 Overview

This introductory assignment covers fundamental RL concepts and tabular methods, establishing the foundation for deep reinforcement learning.

## 📂 Contents

```
HW1/
├── RL_HW1.pdf                  # Assignment questions
├── RL_HW1_Solution.pdf         # Complete solutions
├── SP24_RL_HW1.zip            # Assignment package
└── SP24_RL_HW1_Solution.zip   # Solution package
```

## 🎯 Learning Objectives

- ✅ Master MDP formulation and Bellman equations
- ✅ Implement dynamic programming algorithms
- ✅ Understand temporal difference learning
- ✅ Apply tabular RL methods to GridWorld
- ✅ Analyze convergence properties

## 📚 Topics Covered

### Part 1: Theoretical Foundations

**Markov Decision Processes:**

- State space, action space, transitions
- Reward function and discount factor
- Value functions (V and Q)
- Bellman equations

**Policy Evaluation and Improvement:**

- Policy iteration algorithm
- Value iteration algorithm
- Convergence guarantees

### Part 2: Tabular Methods

**Monte Carlo Methods:**

- First-visit vs every-visit MC
- MC policy evaluation
- MC control with exploring starts

**Temporal Difference Learning:**

- TD(0) prediction
- SARSA (on-policy TD control)
- Q-Learning (off-policy TD control)
- n-step TD methods

### Part 3: Implementation

**GridWorld Environment:**

```python
class GridWorld:
    # States: Grid positions
    # Actions: {up, down, left, right}
    # Rewards: Goal (+1), obstacle (-1), step (0)
    # Dynamics: Deterministic or stochastic transitions
```

**Key Algorithms to Implement:**

1. Value Iteration
2. Policy Iteration
3. Q-Learning
4. SARSA

## 📝 Assignment Structure

### Theoretical Questions (30%)

- MDP formulation problems
- Bellman equation derivations
- Policy and value function analysis
- Convergence proofs

### Implementation (50%)

- Dynamic programming algorithms
- TD learning methods
- Experiments on GridWorld
- Performance comparisons

### Analysis (20%)

- Learning curves and convergence
- Algorithm comparison
- Hyperparameter sensitivity
- Discussion of results

## 💡 Key Concepts

**Value Iteration:**

```
Initialize V(s) = 0
Repeat:
    V(s) ← max_a ∑_{s',r} p(s',r|s,a)[r + γV(s')]
Until converged
Extract policy: π(s) = argmax_a Q(s,a)
```

**Q-Learning:**

```
Initialize Q(s,a) = 0
For each episode:
    For each step:
        a = ε-greedy(s, Q)
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

## 📖 References

- **Sutton & Barto (2018)** - Chapters 3-6
- [Berkeley CS285 Lecture Notes](http://rail.eecs.berkeley.edu/deeprlcourse/)

## ⏱️ Time Estimate

- Reading & Theory: 4-6 hours
- Implementation: 6-10 hours
- Experiments & Analysis: 3-5 hours
- **Total**: 13-21 hours

---

**Difficulty**: ⭐⭐☆☆☆ (Foundational)  
**Prerequisites**: Probability, linear algebra, Python  
**Key Skills**: MDP analysis, dynamic programming, TD learning

