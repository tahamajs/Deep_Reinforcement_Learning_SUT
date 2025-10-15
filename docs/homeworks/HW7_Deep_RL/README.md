# HW7: Deep Reinforcement Learning Fundamentals

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Theory](https://img.shields.io/badge/Type-Theory-yellow.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment focuses on theoretical foundations and fundamental concepts of deep reinforcement learning, including function approximation, convergence guarantees, and the challenges that arise when combining deep learning with RL.

## 🎯 Learning Objectives

1. **Function Approximation**: Understand linear and non-linear function approximators
2. **Deadly Triad**: Learn about the challenges in off-policy learning with function approximation
3. **Convergence Guarantees**: Study conditions for convergence of RL algorithms
4. **Deep Learning in RL**: Understand unique challenges when using neural networks
5. **Stability and Divergence**: Analyze causes of instability in deep RL
6. **Theoretical Foundations**: Master key theorems and proofs in RL theory

## 📂 Directory Structure

```
HW7_Deep_RL/
├── code/                    # (No code for this assignment)
├── answers/                 # (Submit written answers)
├── reports/
│   └── HW7_Questions.pdf   # Theoretical questions
└── README.md
```

## 📚 Core Theoretical Concepts

### 1. Function Approximation in RL

**Why Function Approximation?**
- Tabular methods don't scale to large/continuous state spaces
- Need generalization across similar states
- Memory and computational efficiency

**Types:**

#### Linear Function Approximation
```
V̂(s; w) = ∑i wi φi(s) = wᵀφ(s)

where φ(s) is feature vector
```

**Properties:**
- Convergence guarantees for many algorithms
- Limited representational power
- Easy to analyze theoretically

#### Non-Linear (Deep) Function Approximation
```
V̂(s; θ) = fθ(s)  where fθ is neural network
```

**Properties:**
- High representational power
- Can learn features automatically
- Convergence more difficult to guarantee
- Risk of instability

### 2. The Deadly Triad

**Three Conditions that Together Cause Instability:**

1. **Function Approximation**: Using approximators instead of tabular representations
2. **Bootstrapping**: Updating estimates based on other estimates (TD learning)
3. **Off-Policy Learning**: Learning about policy π while following policy μ ≠ π

**Why Deadly:**
- Any two together usually fine
- All three together can cause divergence
- Fundamental challenge in deep RL

**Classic Example: Baird's Counterexample**
```
Q-learning with linear function approximation
+ Off-policy updates
→ Q-values diverge to infinity
```

**Solutions:**
- Remove one element of triad (often off-policy aspect)
- Use special algorithms (e.g., DQN with target networks)
- Gradient TD methods (GTD, TDC)
- Restriction to on-policy learning

### 3. Convergence Conditions

**Robbins-Monro Conditions for Stochastic Approximation:**

For update rule: θt+1 = θt + αt[target - current]

Convergence requires:
```
1. ∑t αt = ∞        (infinite total learning)
2. ∑t αt² < ∞       (decreasing noise contribution)

Example: αt = 1/t satisfies both
```

**Contraction Mapping Theorem:**
- Bellman operators are contractions (in max norm)
- Guarantees unique fixed point (optimal value function)
- Iterative application converges to fixed point

**When RL Algorithms Converge:**

| Algorithm | Tabular | Linear FA | Non-Linear FA |
|-----------|---------|-----------|---------------|
| **Monte Carlo** | ✅ | ✅ | ✅ (with conditions) |
| **TD(0) On-Policy** | ✅ | ✅ | ⚠️ (often works) |
| **Q-Learning** | ✅ | ❌ (can diverge) | ❌ (needs tricks) |
| **SARSA** | ✅ | ✅ | ⚠️ (often works) |
| **DQN** | ✅ | ✅ | ✅ (with target network) |

### 4. Challenges in Deep RL

#### a) Non-Stationarity
```
Target for supervised learning: fixed labels
Target for RL: Q(s',a') depends on current network

Result: "chasing a moving target"
```

**Solution:** Target networks (DQN)

#### b) Correlated Samples
```
Sequential data: (st, at, rt, st+1) highly correlated
Neural networks assume i.i.d. data

Result: Overfitting to recent experience
```

**Solution:** Experience replay

#### c) High Variance
```
Policy gradient estimates have high variance
Slows learning, causes instability
```

**Solutions:** Baselines, advantage functions, multiple workers

#### d) Credit Assignment
```
Which action in sequence led to reward?
Long-term dependencies difficult to learn
```

**Solutions:** Value functions, eligibility traces, attention mechanisms

#### e) Catastrophic Forgetting
```
Network forgets how to solve old states
when learning new states
```

**Solutions:** Experience replay, regularization, progressive neural networks

### 5. Key Theorems and Results

**Policy Gradient Theorem:**
```
∇θJ(θ) = 𝔼π[∇θ log πθ(a|s) Qπ(s,a)]
```

**Importance:** Enables policy optimization without knowing dynamics

**Policy Improvement Theorem:**
```
If πθ' ≥ πθ for all states (greedy improvement),
then Vπθ'(s) ≥ Vπθ(s) for all states
```

**Importance:** Guarantees policy iteration converges to optimal policy

**Bellman Optimality Equations:**
```
V*(s) = max[R(s,a) + γ ∑s' P(s'|s,a)V*(s')]
         a

Q*(s,a) = R(s,a) + γ ∑s' P(s'|s,a) max Q*(s',a')
                               a'
```

**Importance:** Characterizes optimal value functions

**Approximation in Value Space:**
```
|| V - V* ||∞ ≤ (2γ/(1-γ)) || V - ΠV* ||∞

where Π is projection operator
```

**Importance:** Bounds suboptimality due to function approximation error

## 📊 Topics Covered

1. **Markov Decision Processes**
   - Formal definitions
   - Bellman equations
   - Optimality conditions

2. **Dynamic Programming**
   - Policy iteration
   - Value iteration
   - Asynchronous DP

3. **Temporal Difference Learning**
   - TD(0), TD(λ)
   - Eligibility traces
   - Forward vs backward view

4. **Function Approximation**
   - Linear methods
   - Gradient descent
   - Semi-gradient methods

5. **Deep Q-Networks**
   - Architecture design
   - Training stability
   - Variants and improvements

6. **Policy Optimization**
   - Policy gradient theorem
   - Natural gradients
   - Trust region methods

7. **Exploration**
   - Multi-armed bandits
   - Upper confidence bounds
   - Thompson sampling

8. **Theoretical Analysis**
   - Sample complexity
   - Regret bounds
   - PAC guarantees

## 📖 Key References

### Books

1. **Sutton & Barto (2018)** - *Reinforcement Learning: An Introduction* (2nd ed.)
   - Chapters 9-12 on function approximation
   - Chapter 11 on off-policy methods
   - [Free Online](http://incompleteideas.net/book/the-book-2nd.html)

2. **Szepesvári, C. (2010)** - *Algorithms for Reinforcement Learning*
   - Concise mathematical treatment
   - [Free Online](https://sites.ualberta.ca/~szepesva/RLBook.html)

### Papers

1. **Tsitsiklis, J. N., & Van Roy, B. (1997)**
   - "An analysis of temporal-difference learning with function approximation"
   - IEEE TAC

2. **Baird, L. (1995)**
   - "Residual algorithms: Reinforcement learning with function approximation"
   - ICML (shows divergence example)

3. **Mnih, V., et al. (2015)**
   - "Human-level control through deep reinforcement learning"
   - Nature (DQN paper)

4. **Sutton, R. S., et al. (2009)**
   - "Fast gradient-descent methods for temporal-difference learning with linear function approximation"
   - ICML (GTD algorithms)

### Surveys

1. **François-Lavet, V., et al. (2018)** - "An Introduction to Deep Reinforcement Learning" - arXiv:1811.12560

2. **Li, Y. (2017)** - "Deep Reinforcement Learning: An Overview" - arXiv:1701.07274

## 💡 Discussion Questions

1. **Why does Q-learning with linear function approximation sometimes diverge?**
   
2. **How do target networks help stabilize deep Q-learning?**

3. **What is the relationship between the deadly triad and DQN's design choices?**

4. **Why does policy gradient have theoretical advantages over value-based methods in some settings?**

5. **How does experience replay address non-stationarity in deep RL?**

6. **What are the trade-offs between on-policy and off-policy learning?**

7. **Why is credit assignment harder in RL than supervised learning?**

8. **How do eligibility traces relate to n-step methods?**

## 📝 Assignment Format

### Question Types

1. **Conceptual Questions**: Explain key concepts and their relationships
2. **Mathematical Derivations**: Prove convergence, derive updates
3. **Analysis Questions**: Compare algorithms, analyze scenarios
4. **Design Questions**: Propose solutions to given problems

### Suggested Approach

1. **Review Lecture Notes**: Understand core concepts
2. **Read Sutton & Barto**: Chapters 9-13 essential
3. **Work Through Examples**: Practice derivations
4. **Discuss with Peers**: Clarify confusing topics
5. **Write Clear Answers**: Use mathematical notation properly

### Grading Rubric

- **Correctness (60%)**: Accurate answers and derivations
- **Clarity (20%)**: Clear explanations and notation
- **Completeness (15%)**: Address all parts of questions
- **Insight (5%)**: Demonstrate deep understanding

## 🎓 Study Tips

### For Mathematical Questions

- Review linear algebra (vectors, matrices, norms)
- Understand expectation and probability
- Practice writing formal proofs
- Check dimensional consistency

### For Conceptual Questions

- Use diagrams to illustrate concepts
- Provide concrete examples
- Connect to algorithms you've implemented
- Reference seminal papers

### For Comparison Questions

- Create comparison tables
- Identify trade-offs
- Consider different scenarios
- Support with theoretical results

## 🔗 Additional Resources

- **OpenAI Spinning Up**: [Deep RL Theory](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- **David Silver's Course**: [UCL RL Lectures](https://www.davidsilver.uk/teaching/)
- **Berkeley CS285**: [Deep RL Course](http://rail.eecs.berkeley.edu/deeprlcourse/)
- **DeepMind x UCL**: [RL Lecture Series](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)

## ⚠️ Common Mistakes to Avoid

1. Confusing value functions V(s) and Q(s,a)
2. Ignoring discount factor γ in derivations
3. Mixing on-policy and off-policy update rules
4. Incorrect use of expectations and sampling
5. Forgetting conditions for convergence theorems

---

**Course:** Deep Reinforcement Learning  
**Assignment Type:** Written/Theoretical  
**Last Updated:** 2024

For questions, consult course staff during office hours or discussion forums.
