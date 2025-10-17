# HW0: Introduction to Reinforcement Learning

## 📋 Overview

This introductory assignment covers the fundamental concepts of Reinforcement Learning, establishing the theoretical foundation for the course.

## 📂 Contents

```
HW0/
├── RL_HW0.pdf           # Problem set
├── SP23_RL_HW0.zip      # Complete package
└── README.md            # This file
```

## 🎯 Learning Objectives

By completing this assignment, you will:

- ✅ Understand Markov Decision Processes (MDPs)
- ✅ Master Bellman equations for value functions
- ✅ Analyze policy evaluation and improvement
- ✅ Work with discount factors and return calculations
- ✅ Understand the relationship between policies and value functions

## 📚 Topics Covered

### 1. Markov Decision Processes (MDPs)

- **States (S)**: Set of all possible states
- **Actions (A)**: Set of available actions
- **Transitions (P)**: State transition probabilities
- **Rewards (R)**: Reward function
- **Discount Factor (γ)**: Future reward discounting

**MDP Tuple:** (S, A, P, R, γ)

### 2. Value Functions

**State Value Function:**

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[∑(γ^k * R_{t+k+1}) | S_t = s]
```

**Action Value Function (Q-function):**

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

### 3. Bellman Equations

**Bellman Expectation Equation:**

```
V^π(s) = ∑_a π(a|s) * ∑_{s',r} p(s',r|s,a)[r + γV^π(s')]
```

**Bellman Optimality Equation:**

```
V*(s) = max_a ∑_{s',r} p(s',r|s,a)[r + γV*(s')]
```

### 4. Policies

**Deterministic Policy:** π(s) → a

**Stochastic Policy:** π(a|s) → [0, 1]

**Optimal Policy:** π\* = argmax_π V^π(s) for all s

### 5. Returns and Discounting

**Finite Horizon Return:**

```
G_t = R_{t+1} + R_{t+2} + ... + R_T
```

**Infinite Horizon Discounted Return:**

```
G_t = R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ...
    = ∑_{k=0}^∞ γ^k R_{t+k+1}
```

## 📝 Problem Types

The assignment typically includes:

### Theoretical Questions

1. **MDP Formulation**: Defining MDPs for given scenarios
2. **Bellman Derivations**: Proving properties of value functions
3. **Policy Analysis**: Comparing different policies
4. **Return Calculations**: Computing expected returns

### Computational Problems

1. **Value Iteration**: Manual iterations on small MDPs
2. **Policy Evaluation**: Computing V^π for given policies
3. **Optimal Policy**: Finding π\* through analysis
4. **Discount Factor Effects**: Analyzing γ impact

## 🔑 Key Concepts

### Why MDPs?

MDPs provide a mathematical framework for sequential decision-making under uncertainty, capturing the essential elements of RL problems.

### The Markov Property

```
P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0)
= P(S_{t+1} | S_t, A_t)
```

The future is independent of the past given the present.

### Policy vs Value

- **Policy**: What to do (action selection strategy)
- **Value**: How good it is (expected return)
- **Relationship**: Optimal policy achieves highest value

### Discount Factor (γ)

- **γ = 0**: Myopic (only immediate reward)
- **γ = 1**: Farsighted (all future rewards equal)
- **0 < γ < 1**: Balanced (prefer nearer rewards)

## 💡 Problem-Solving Tips

### For Bellman Equations

1. Start with the definition of value function
2. Expand expectation over actions (policy)
3. Expand expectation over next states (dynamics)
4. Apply recursive structure

### For Value Iteration

1. Initialize V(s) = 0 for all states
2. Update: V(s) ← max_a ∑ p(s'|s,a)[r + γV(s')]
3. Repeat until convergence
4. Extract policy: π(s) = argmax_a Q(s,a)

### For Policy Evaluation

1. Fix policy π
2. Solve system of linear equations:
   ```
   V^π(s) = ∑_a π(a|s) ∑ p(s'|s,a)[r + γV^π(s')]
   ```
3. Or iterate until convergence

## 📖 Reference Materials

### Sutton & Barto (2018)

- **Chapter 3**: Finite MDPs
- **Chapter 4**: Dynamic Programming
- Sections 3.1-3.6 are essential

### Additional Resources

- [UCL RL Course Lecture 2](https://www.davidsilver.uk/teaching/) - MDPs
- [Stanford CS234 Lecture 2](http://web.stanford.edu/class/cs234/) - MDP Formulation

## 🧮 Example Problem Walkthrough

### Problem: GridWorld MDP

**Setup:**

- 3x3 grid
- Start: (0,0)
- Goal: (2,2), reward = +1
- Other states: reward = 0
- Actions: {up, down, left, right}
- Deterministic transitions
- γ = 0.9

**Question:** Compute V\*(s) for all states.

**Solution Approach:**

1. **Initialize:** V(s) = 0 for all s
2. **Iterate:**
   ```
   V(2,2) = 1  (terminal state)
   V(1,2) = 0 + 0.9 * 1 = 0.9
   V(2,1) = 0 + 0.9 * 1 = 0.9
   V(1,1) = 0 + 0.9 * 0.9 = 0.81
   ...
   ```
3. **Converge:** Continue until changes < threshold

## ✅ Self-Check Questions

Before submitting, ensure you can answer:

1. ☐ What makes a process Markovian?
2. ☐ Why do we need discount factors?
3. ☐ What's the difference between V and Q functions?
4. ☐ How do Bellman equations enable DP?
5. ☐ What defines an optimal policy?
6. ☐ How does policy relate to value function?
7. ☐ What happens when γ → 0? γ → 1?
8. ☐ Can multiple policies be optimal?

## 🎓 Expected Time

- **Reading Assignment**: 2-3 hours
- **Problem Set**: 4-6 hours
- **Review & Writeup**: 1-2 hours
- **Total**: 7-11 hours

## 📊 Grading Breakdown

Typical grading:

- MDP Formulation (20%)
- Bellman Equations (25%)
- Value Computations (30%)
- Policy Analysis (25%)

## 🚀 Getting Started

1. **Read Sutton & Barto Chapter 3**
2. **Review lecture slides on MDPs**
3. **Open RL_HW0.pdf**
4. **Attempt problems sequentially**
5. **Check your understanding with solutions**

## 💬 Common Mistakes

❌ **Mistake 1**: Forgetting discount factor in return calculations
✅ **Fix**: Always include γ^k for k-step ahead rewards

❌ **Mistake 2**: Confusing policy π(a|s) with value V(s)
✅ **Fix**: Policy is action distribution; value is expected return

❌ **Mistake 3**: Not checking Markov property
✅ **Fix**: Verify state contains all decision-relevant information

❌ **Mistake 4**: Wrong expectation order in Bellman equations
✅ **Fix**: First over actions (policy), then over states (dynamics)

## 📚 Key Equations Reference

```
Return: G_t = ∑_{k=0}^∞ γ^k R_{t+k+1}

State Value: V^π(s) = E_π[G_t | S_t = s]

Action Value: Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]

Bellman Expectation: V^π(s) = ∑_a π(a|s) Q^π(s,a)

Bellman Optimality: V*(s) = max_a Q*(s,a)

Q-V Relationship: Q^π(s,a) = R(s,a) + γ∑_{s'} P(s'|s,a)V^π(s')
```

## 🔗 Related Topics

This assignment prepares you for:

- **HW1**: Implementing value iteration and policy iteration
- **HW2**: Policy gradient methods
- **Future Topics**: Model-free RL, function approximation

## 📧 Getting Help

- **Office Hours**: Review MDP concepts
- **Discussion Forum**: Post conceptual questions
- **Study Groups**: Work through problems together
- **Textbook**: Sutton & Barto Chapters 3-4

---

**Difficulty**: ⭐⭐☆☆☆ (Foundational)  
**Estimated Time**: 7-11 hours  
**Prerequisites**: Probability, basic linear algebra

Good luck! This foundation is crucial for the entire course.

