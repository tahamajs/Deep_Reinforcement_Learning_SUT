# HW3: Policy Gradient Methods

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Policy-Based](https://img.shields.io/badge/Methods-Policy--Based-orange.svg)](https://www.deepmind.com/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment explores **policy gradient methods**, a class of algorithms that directly optimize the policy by following the gradient of expected return. Unlike value-based methods that derive policies from value functions, policy gradient methods parameterize the policy directly and optimize it using gradient ascent.

## 🎯 Learning Objectives

By completing this assignment, you will:

1. **Understand Policy Parameterization**: Learn how to represent policies with neural networks
2. **Master REINFORCE Algorithm**: Implement the foundational policy gradient method
3. **Variance Reduction with Baselines**: Understand and implement baseline techniques to reduce gradient variance
4. **Compare with Value-Based Methods**: Analyze trade-offs between policy gradient and Q-learning approaches
5. **Handle Continuous Actions**: Learn how policy gradients naturally extend to continuous action spaces
6. **Understand Credit Assignment**: Study how gradients attribute credit to actions

## 📂 Directory Structure

```
HW3_Policy_Gradients/
├── code/
│   ├── HW3_P1_REINFORCE_VS_GA.ipynb              # REINFORCE vs Genetic Algorithms
│   ├── HW3_P2_CartPole_REINFORCE_Baseline.ipynb  # Baseline variance reduction
│   ├── HW3_P3_MountainCarContinuous_REINFORCE.ipynb  # Continuous actions
│   └── HW3_P4_REINFORCEvsDQN.ipynb                # Policy vs Value comparison
├── answers/
│   ├── HW3_P1_REINFORCE_VS_GA_Solution.ipynb
│   ├── HW3_P2_CartPole_REINFORCE_Baseline_Solution.ipynb
│   ├── HW3_P3_MountainCarContinuous_REINFORCE_Solution.ipynb
│   ├── HW3_P4_REINFORCEvsDQN_Solution.ipynb
│   └── HW3_Solution.pdf
├── reports/
│   └── HW3_Questions.pdf
└── README.md
```

## 📚 Theoretical Background

### 1. Policy Gradient Theorem

The **Policy Gradient Theorem** provides the foundation for all policy gradient methods. It shows how to compute the gradient of expected return with respect to policy parameters.

**Objective:**

```
J(θ) = 𝔼τ~πθ[∑t γᵗ r(st, at)]
```

**Policy Gradient:**

```
∇θ J(θ) = 𝔼τ~πθ[∑t ∇θ log πθ(at|st) Gt]

where Gt = ∑k=t^T γᵏ⁻ᵗ rk (return from time t)
```

**Key Insight:** We can estimate gradients using samples, without knowing environment dynamics!

**Intuition:**

- `∇θ log πθ(at|st)`: Direction that increases probability of action at
- `Gt`: How good was this action (return after taking it)
- Together: Increase probability of actions that led to high returns

### 2. REINFORCE Algorithm

**REINFORCE** (REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility) is the Monte Carlo policy gradient algorithm.

**Algorithm:**

```python
Initialize policy parameters θ
for each episode:
    Generate episode τ = (s0, a0, r1, ..., sT) using πθ
    for t = 0 to T-1:
        Gt = ∑k=t^T γᵏ⁻ᵗ rk
        θ ← θ + α ∇θ log πθ(at|st) Gt
```

**Properties:**

- **Unbiased**: Gradient estimate is correct in expectation
- **High Variance**: Monte Carlo estimates have high variance
- **On-Policy**: Must sample from current policy
- **Sample Inefficient**: Requires many episodes

**Why It Works:**
The log-derivative trick converts policy gradient into expectation over trajectories:

```
∇θ 𝔼[R] = ∇θ ∑τ P(τ|θ)R(τ)
         = ∑τ P(τ|θ) ∇θ log P(τ|θ) R(τ)
         = 𝔼τ[∇θ log P(τ|θ) R(τ)]
```

### 3. Variance Reduction with Baselines

**Problem:** High variance in gradient estimates leads to slow, unstable learning.

**Solution:** Subtract a baseline b(st) from returns without introducing bias:

```
∇θ J(θ) = 𝔼[∑t ∇θ log πθ(at|st) (Gt - b(st))]
```

**Why Baselines Work:**

```
𝔼[∇θ log πθ(at|st) b(st)] = ∑a πθ(a|st) ∇θ log πθ(a|st) b(st)
                            = b(st) ∇θ ∑a πθ(a|st)
                            = b(st) ∇θ 1 = 0
```

**Common Baselines:**

#### a) Constant Baseline

```
b(st) = 𝔼[Gt]  (average return)
```

#### b) Value Function Baseline

```
b(st) = V(st)  (state value function)
```

This is optimal in terms of variance reduction!

#### c) Advantage Function

```
A(st, at) = Q(st, at) - V(st) = Gt - V(st)
```

**Advantage Interpretation:**

- Positive: Action is better than average
- Negative: Action is worse than average
- Zero: Action is average

### 4. Policy Parameterization

#### Discrete Actions (Softmax Policy)

```python
class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)
```

**Log Probability:**

```python
log_prob = torch.log(probs[action] + 1e-8)
```

#### Continuous Actions (Gaussian Policy)

```python
class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
```

**Why Gaussian?**

- Naturally handles continuous actions
- Easy to compute log probabilities
- Adjustable exploration via std

### 5. REINFORCE vs Value-Based Methods

| Aspect                  | REINFORCE       | DQN                         |
| ----------------------- | --------------- | --------------------------- |
| **What is learned**     | Policy πθ(a\|s) | Value Q(s,a)                |
| **Action selection**    | Sample from πθ  | argmax Q                    |
| **Continuous actions**  | Natural         | Difficult                   |
| **Convergence**         | Local optimum   | Global optimum (in tabular) |
| **Sample efficiency**   | Lower           | Higher                      |
| **Variance**            | Higher          | Lower                       |
| **Stochastic policies** | Yes             | No (except ε-greedy)        |

**When to use Policy Gradients:**

- Continuous action spaces
- Stochastic policies needed
- High-dimensional action spaces
- Non-differentiable policies

**When to use Value-Based:**

- Discrete actions
- Deterministic policies
- Sample efficiency critical
- Off-policy learning needed

### 6. Genetic Algorithms vs REINFORCE

**Genetic Algorithms (GA):**

- Population-based evolutionary approach
- No gradient information used
- Random mutations and selection
- Parallel evaluation

**Comparison:**

- **GA**: Derivative-free, highly parallel, but sample inefficient
- **REINFORCE**: Uses gradient information, more sample efficient
- **GA**: Works with non-differentiable policies
- **REINFORCE**: Requires differentiable policy

## 💻 Implementation Details

### Part 1: REINFORCE vs Genetic Algorithms

**Objective:** Compare gradient-based (REINFORCE) with evolutionary (GA) approaches.

**Tasks:**

1. Implement basic REINFORCE algorithm
2. Implement genetic algorithm with mutation and crossover
3. Compare sample efficiency and final performance
4. Analyze convergence speed

**Expected Observations:**

- REINFORCE converges faster with fewer samples
- GA requires larger population but is more parallelizable
- REINFORCE more stable with proper hyperparameters

### Part 2: CartPole with Baseline

**Objective:** Demonstrate variance reduction using baselines.

**Tasks:**

1. Implement REINFORCE without baseline
2. Implement value function baseline (V(s))
3. Compare training stability and variance
4. Visualize gradient variance across training

**Baseline Network:**

```python
class ValueBaseline(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)
```

**Training Loop:**

```python
# Policy loss
advantages = returns - baseline_values
policy_loss = -(log_probs * advantages).mean()

# Baseline loss
baseline_loss = F.mse_loss(baseline_values, returns)

# Total loss
loss = policy_loss + baseline_loss
```

**Expected Results:**

- 50-70% variance reduction with baseline
- Faster convergence
- More stable training

### Part 3: MountainCar Continuous

**Objective:** Apply REINFORCE to continuous action space.

**Tasks:**

1. Implement Gaussian policy for continuous actions
2. Handle action bounds and scaling
3. Tune exploration (std parameter)
4. Analyze policy learned

**Gaussian Policy:**

```python
# Forward pass
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action)

# Gradient computation
loss = -(log_prob * returns).mean()
```

**Challenges:**

- MountainCar has sparse rewards
- Requires good exploration
- May need reward shaping

### Part 4: REINFORCE vs DQN

**Objective:** Direct comparison of policy-based and value-based methods.

**Tasks:**

1. Implement both REINFORCE and DQN on same environment
2. Compare sample efficiency
3. Compare final performance
4. Analyze learned policies (stochastic vs deterministic)

**Metrics to Compare:**

- Episodes to convergence
- Final average reward
- Training stability (variance across runs)
- Policy entropy (exploration)

## 📊 Evaluation Metrics

1. **Return**: Average episode return
2. **Return Variance**: Measure training stability
3. **Gradient Variance**: Track with and without baseline
4. **Policy Entropy**: Measure exploration
   ```
   H(π) = -∑a πθ(a|s) log πθ(a|s)
   ```
5. **Baseline Error**: MSE between V(s) and actual returns
6. **Convergence Speed**: Episodes to reach threshold

## 🔧 Requirements

```python
# Core Libraries
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# RL Environment
gymnasium>=0.28.0

# Deep Learning
torch>=2.0.0

# Utilities
pandas>=1.3.0
tqdm>=4.62.0
scipy>=1.7.0  # For genetic algorithms
```

## 🚀 Getting Started

### Installation

```bash
cd HW3_Policy_Gradients
pip install -r requirements.txt
```

### Running Experiments

```bash
# Part 1: REINFORCE vs GA
jupyter notebook code/HW3_P1_REINFORCE_VS_GA.ipynb

# Part 2: Baseline comparison
jupyter notebook code/HW3_P2_CartPole_REINFORCE_Baseline.ipynb

# Part 3: Continuous actions
jupyter notebook code/HW3_P3_MountainCarContinuous_REINFORCE.ipynb

# Part 4: REINFORCE vs DQN
jupyter notebook code/HW3_P4_REINFORCEvsDQN.ipynb
```

## 📈 Expected Results

### CartPole

- **Without Baseline**: Converges in ~500-1000 episodes, high variance
- **With Baseline**: Converges in ~300-500 episodes, lower variance
- **Final Performance**: 195+ average reward

### MountainCar Continuous

- **Challenge**: Sparse rewards, exploration critical
- **Convergence**: 200-500 episodes with good hyperparameters
- **Final Performance**: 90+ average reward

### REINFORCE vs DQN

- **DQN**: Faster convergence, more sample efficient
- **REINFORCE**: Naturally handles stochastic policies
- **Performance**: Similar final performance, different learning curves

## 🐛 Common Issues & Solutions

### Issue 1: Policy Collapses (No Exploration)

**Symptoms:** Policy becomes deterministic early, stops learning
**Solutions:**

- Add entropy bonus to loss: `loss = policy_loss - β * entropy`
- Use larger initial std for Gaussian policies
- Ensure sufficient exploration episodes

### Issue 2: High Variance, Unstable Training

**Symptoms:** Reward fluctuates wildly, doesn't converge
**Solutions:**

- Implement baseline
- Normalize returns: `returns = (returns - mean) / (std + 1e-8)`
- Reduce learning rate
- Increase batch size (more episodes per update)

### Issue 3: Gradients Vanishing/Exploding

**Symptoms:** Loss becomes NaN or doesn't change
**Solutions:**

- Clip gradients: `torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)`
- Check log probabilities for numerical issues (add small epsilon)
- Normalize advantages

### Issue 4: MountainCar Not Learning

**Symptoms:** Agent doesn't reach goal
**Solutions:**

- Implement reward shaping (penalize for time, reward for height)
- Increase exploration (larger std)
- Use longer episodes (increase max_steps)
- Consider using curriculum learning

## 📖 References

### Seminal Papers

1. **Williams, R. J. (1992)**

   - "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
   - _Machine Learning, 8_(3-4), 229-256
   - [Paper](https://link.springer.com/article/10.1007/BF00992696)

2. **Sutton, R. S., et al. (2000)**

   - "Policy gradient methods for reinforcement learning with function approximation"
   - _NIPS_
   - [Paper](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

3. **Greensmith, E., Bartlett, P. L., & Baxter, J. (2004)**
   - "Variance reduction techniques for gradient estimates in reinforcement learning"
   - _Journal of Machine Learning Research, 5_, 1471-1530

### Books & Tutorials

1. **Sutton & Barto (2018)** - Chapter 13: Policy Gradient Methods
2. **Spinning Up in Deep RL** - [Policy Gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
3. **CS285 Berkeley** - [Policy Gradients Lecture](http://rail.eecs.berkeley.edu/deeprlcourse/)

## 💡 Discussion Questions

1. **Why do policy gradients have high variance compared to value-based methods?**
2. **How does a baseline reduce variance without introducing bias?**

3. **When would you prefer policy gradients over value-based methods?**

4. **What is the relationship between policy gradient and supervised learning?**

5. **How do policy gradients handle continuous action spaces?**

6. **What are the trade-offs between on-policy and off-policy methods?**

## 🎓 Extensions & Challenges

### Easy

- Implement entropy regularization
- Try different baseline functions
- Visualize policy evolution

### Medium

- Implement advantage actor-critic (A2C)
- Add importance sampling for off-policy learning
- Implement natural policy gradients

### Hard

- Implement Trust Region Policy Optimization (TRPO)
- Add generalized advantage estimation (GAE)
- Apply to high-dimensional environments

## 📝 Assignment Deliverables

1. **Code**: All 4 Jupyter notebooks with executed cells
2. **Report**:
   - Comparison plots (with/without baseline, REINFORCE vs DQN)
   - Variance analysis
   - Discussion of results
3. **Experiments**: Multiple runs with different seeds, error bars on plots

### Grading Rubric

- **Implementation (40%)**: Correct REINFORCE and baseline implementation
- **Analysis (30%)**: Insightful comparison and variance analysis
- **Experiments (20%)**: Comprehensive evaluation across environments
- **Presentation (10%)**: Clear plots and explanations

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024

For questions, contact course staff or open an issue in the repository.
