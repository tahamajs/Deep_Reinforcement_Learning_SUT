# HW13: Offline Reinforcement Learning - Complete Solutions

**Course:** Deep Reinforcement Learning  
**Assignment:** Homework 13 - Offline RL  
**Format:** IEEE Standard  
**Date:** October 15, 2025

---

## Table of Contents

1. [Introduction to Offline Reinforcement Learning](#1-introduction)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [The Distributional Shift Problem](#3-distributional-shift)
4. [Conservative Q-Learning (CQL)](#4-conservative-q-learning)
5. [Implicit Q-Learning (IQL)](#5-implicit-q-learning)
6. [Batch-Constrained Deep Q-Learning (BCQ)](#6-batch-constrained-q-learning)
7. [Model-Based Offline RL (MOPO)](#7-model-based-offline-rl)
8. [Implementation Details](#8-implementation)
9. [Experimental Results and Analysis](#9-experiments)
10. [Real-World Applications](#10-applications)
11. [Discussion and Future Work](#11-discussion)
12. [References](#12-references)

---

## 1. Introduction to Offline Reinforcement Learning

### 1.1 Problem Statement

**Question 1.1:** What is Offline Reinforcement Learning, and how does it differ from traditional RL?

**Answer:**

Offline Reinforcement Learning (also known as Batch RL or Off-policy RL from fixed datasets) is a paradigm where an agent learns an optimal policy from a fixed dataset of interactions without any additional environment interaction during training.

**Key Differences from Traditional RL:**

| Aspect                 | Traditional RL                 | Offline RL                      |
| ---------------------- | ------------------------------ | ------------------------------- |
| **Data Collection**    | Online, interactive            | Fixed dataset, no interaction   |
| **Exploration**        | Active exploration possible    | No exploration, fixed data      |
| **Policy Improvement** | Iterative with environment     | One-shot from dataset           |
| **Risk**               | Can cause harm during training | Safe, no environmental impact   |
| **Applications**       | Simulations, games             | Healthcare, autonomous vehicles |

**Mathematical Formulation:**

Given a fixed dataset \(D = \{(s*t, a_t, r_t, s*{t+1})\}_{t=1}^N\) collected by some behavior policy \(\pi_\beta\), the goal is to learn a policy \(\pi^\*\) that maximizes:

\[
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]
\]

without any additional environment interaction.

---

### 1.2 Motivation and Applications

**Question 1.2:** Why is Offline RL important, and what are its real-world applications?

**Answer:**

**Motivation:**

1. **Safety:** In critical domains (healthcare, autonomous driving), online exploration can be dangerous or unethical
2. **Cost:** Environment interaction can be expensive (robotics experiments, clinical trials)
3. **Data Availability:** Large datasets often exist from previous systems (logs, historical records)
4. **Regulation:** Some domains prohibit exploration on real systems (financial trading, medical treatments)

**Real-World Applications:**

1. **Healthcare:**

   - Learning treatment policies from electronic health records (EHRs)
   - Optimizing drug dosing from historical patient data
   - Personalized medicine without patient risk

2. **Autonomous Driving:**

   - Learning driving policies from human demonstration data
   - Avoiding dangerous exploration on real roads
   - Utilizing fleet data for policy improvement

3. **Robotics:**

   - Learning manipulation from demonstrations
   - Avoiding hardware wear from exploration
   - Transfer from simulation to real robots

4. **Recommender Systems:**

   - Learning from logged user interactions
   - Improving recommendations without A/B testing
   - Handling large-scale user data

5. **Finance:**
   - Trading strategies from historical market data
   - Risk management without real capital exposure
   - Regulatory compliance

---

## 2. Theoretical Foundations

### 2.1 Markov Decision Process

**Question 2.1:** Define the Markov Decision Process (MDP) framework for Offline RL.

**Answer:**

An Offline RL problem is defined by a tuple \(\langle \mathcal{S}, \mathcal{A}, P, r, \gamma, D \rangle\):

- **\(\mathcal{S}\)**: State space
- **\(\mathcal{A}\)**: Action space
- **\(P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]\)**: Transition dynamics
- **\(r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}\)**: Reward function
- **\(\gamma \in [0,1)\)**: Discount factor
- **\(D\)**: Fixed dataset collected by behavior policy \(\pi\_\beta\)

**Key Assumptions:**

1. **Markov Property:** \(P(s*{t+1}|s_t, a_t, s*{t-1}, ...) = P(s\_{t+1}|s_t, a_t)\)
2. **Stationary:** Dynamics and rewards don't change over time
3. **Incomplete Coverage:** Dataset may not cover all state-action pairs

**Objective:**

Find policy \(\pi^\*\) that maximizes expected return:

\[
\pi^\* = \arg\max*{\pi} J(\pi) = \arg\max*{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]
\]

**Constraints:**

- Policy must be learned from \(D\) only
- No environment queries allowed
- Must handle out-of-distribution (OOD) actions

---

### 2.2 Value Functions

**Question 2.2:** Define the value functions used in Offline RL and their Bellman equations.

**Answer:**

**State Value Function:**

\[
V^\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s \right]
\]

**Action Value Function (Q-function):**

\[
Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s, a_0 = a \right]
\]

**Bellman Equations:**

State Value:
\[
V^\pi(s) = \mathbb{E}\_{a \sim \pi} \left[ Q^\pi(s, a) \right]
\]

Action Value:
\[
Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}\_{s' \sim P} \left[ V^\pi(s') \right]
\]

**Bellman Optimality Equations:**

\[
V^_(s) = \max_a Q^_(s, a)
\]

\[
Q^_(s, a) = r(s, a) + \gamma \mathbb{E}*{s' \sim P} \left[ \max*{a'} Q^_(s', a') \right]
\]

**In Offline RL:**

The key challenge is that we cannot compute \(\mathbb{E}\_{s' \sim P}\) directly. We must estimate it from the dataset \(D\), leading to potential errors for OOD actions.

---

## 3. The Distributional Shift Problem

### 3.1 Understanding Distributional Shift

**Question 3.1:** Explain the distributional shift problem in Offline RL and why it causes failure.

**Answer:**

**Definition:**

Distributional shift occurs when the learned policy \(\pi\) encounters state-action pairs that were not well-represented in the training dataset collected by behavior policy \(\pi\_\beta\).

**Mathematical Formulation:**

Let \(d^\pi(s, a)\) be the state-action visitation frequency under policy \(\pi\) and \(d^{\pi\_\beta}(s, a)\) be the visitation frequency under the behavior policy.

**Distributional shift occurs when:**

\[
d^\pi(s, a) \neq d^{\pi\_\beta}(s, a)
\]

**Why It Causes Failure:**

1. **Extrapolation Error:**

   - Q-function is learned from dataset: \(Q*\theta(s, a) \approx Q^{\pi*\beta}(s, a)\) for \((s,a) \in D\)
   - For OOD actions: \(Q\_\theta(s, a)\) is unreliable
   - Policy maximizes \(Q\_\theta\), selecting actions with overestimated values

2. **Bootstrapping Amplification:**

   - Bellman backup: \(Q(s, a) \leftarrow r + \gamma \max\_{a'} Q(s', a')\)
   - Overestimation in one step propagates through bootstrapping
   - Errors compound exponentially with planning horizon

3. **Narrow Data Distribution:**
   - Behavior policy may be suboptimal, covering only limited regions
   - Learned policy tries to exploit, moving to uncovered regions
   - No corrective feedback available

**Example:**

```
Dataset: Expert driving straight on highway
Learned Policy: Tries aggressive lane changes (high estimated Q)
Reality: No data on aggressive maneuvers → crash
```

**Illustration:**

```
Q-values fitted to data (•):

Q  |
   |   •   •
   | •   ?   •    ← ? = OOD action, Q may be overestimated
   |   •   •
   |_____________
      actions

Policy picks argmax_a Q(s,a) → selects OOD action with wrong value
```

---

### 3.2 Quantifying Distributional Shift

**Question 3.2:** How can we measure the severity of distributional shift in Offline RL?

**Answer:**

**1. Dataset Coverage Metrics:**

**Support Coverage:**
\[
C\_{\text{support}} = \frac{|\{(s,a) : d^\pi(s,a) > 0 \land (s,a) \in D\}|}{|\{(s,a) : d^\pi(s,a) > 0\}|}
\]

Measures what fraction of policy's state-actions are in the dataset.

**Density Ratio:**
\[
w(s, a) = \frac{d^\pi(s, a)}{d^{\pi\_\beta}(s, a)}
\]

If \(w(s,a) > 1\): Policy visits more than behavior policy (potential OOD)

**2. Divergence Measures:**

**KL Divergence:**
\[
D*{KL}(d^\pi || d^{\pi*\beta}) = \mathbb{E}_{(s,a) \sim d^\pi} \left[ \log \frac{d^\pi(s,a)}{d^{\pi_\beta}(s,a)} \right]
\]

**Total Variation Distance:**
\[
D*{TV}(d^\pi, d^{\pi*\beta}) = \frac{1}{2} \sum*{s,a} |d^\pi(s,a) - d^{\pi*\beta}(s,a)|
\]

**3. Uncertainty Estimates:**

**Ensemble Disagreement:**

- Train ensemble of Q-functions: \(\{Q*{\theta_i}\}*{i=1}^K\)
- Measure variance: \(\sigma^2(s,a) = \text{Var}_i[Q_{\theta_i}(s,a)]\)
- High variance indicates OOD region

**4. Practical Indicators:**

- **Performance gap:** \(J(\pi) - J(\pi\_\beta)\) (if environment available)
- **In-dataset performance:** Evaluate on held-out offline data
- **Constraint violations:** Check safety/feasibility in logged data

---

## 4. Conservative Q-Learning (CQL)

### 4.1 CQL Theory

**Question 4.1:** Explain the theoretical foundation of Conservative Q-Learning (CQL).

**Answer:**

**Core Idea:**

CQL learns a Q-function that provides a **lower bound** on the true Q-value, preventing overestimation of OOD actions.

**Mathematical Formulation:**

The CQL objective combines a conservative regularizer with the Bellman error:

\[
\min*Q \alpha \left( \mathbb{E}*{s \sim D} \left[ \log \sum_a \exp(Q(s,a)) \right] - \mathbb{E}_{(s,a) \sim D} [Q(s,a)] \right) + \frac{1}{2} \mathbb{E}_{(s,a,s') \sim D} \left[ (Q(s,a) - \mathcal{B}^\pi Q(s,a))^2 \right]
\]

Where:

- **First term (CQL penalty):** Pushes down Q-values for all actions
- **Second term (Negation):** Raises Q-values for dataset actions
- **Third term (Bellman error):** Ensures Q-function satisfies Bellman equation on dataset

**Why It Works:**

1. **Minimizes Value Overestimation:**

   - LogSumExp term encourages low Q-values
   - Dataset term prevents underestimating in-distribution actions
   - Result: Conservative estimates outside dataset

2. **Lower Bound Guarantee:**

Under certain conditions, CQL provides:
\[
Q\_{CQL}(s, a) \leq Q^\pi(s, a) \quad \forall (s,a)
\]

This ensures the learned policy doesn't over-optimistically exploit.

3. **Connection to Importance Sampling:**

The CQL objective approximates:
\[
\max*\pi \mathbb{E}*{(s,a) \sim d^{\pi\_\beta}} \left[ w(s,a) \cdot Q^\pi(s,a) \right]
\]

where \(w(s,a)\) weights down OOD actions.

---

### 4.2 CQL Implementation

**Question 4.2:** Provide a detailed implementation of Conservative Q-Learning.

**Answer:**

**Algorithm: Conservative Q-Learning (CQL)**

```
Input: Dataset D, regularization parameter α, learning rates η_Q, η_π
Initialize: Q-networks Q_θ, Q_θ', policy π_φ, target networks
For episode = 1 to M:
    Sample batch B = {(s, a, r, s')} from D

    # Compute CQL Q-loss
    Q_current = Q_θ(s, a)

    # Bellman target
    with torch.no_grad():
        a' ~ π_φ(s')
        Q_target = r + γ * min(Q_θ'(s', a'))

    # Standard Bellman error
    bellman_error = (Q_current - Q_target)²

    # CQL penalty
    # Sample random actions and current policy actions
    a_rand ~ Uniform(A)
    a_curr ~ π_φ(s)

    # Compute logsumexp over action distribution
    Q_logsumexp = log(∑_a exp(Q_θ(s, a)))
    Q_data = Q_θ(s, a)  # Dataset actions

    cql_loss = α * (Q_logsumexp - Q_data)

    # Total Q-loss
    Q_loss = cql_loss + bellman_error

    # Update Q-network
    θ ← θ - η_Q * ∇_θ Q_loss

    # Update policy (standard SAC)
    π_loss = -E[Q_θ(s, π_φ(s))]
    φ ← φ - η_π * ∇_φ π_loss

    # Update target networks (soft update)
    θ' ← τ*θ + (1-τ)*θ'
```

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return torch.tanh(action), log_prob

class CQL:
    def __init__(self, state_dim, action_dim, alpha=1.0, lr=3e-4):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)
        self.Q1_target = QNetwork(state_dim, action_dim)
        self.Q2_target = QNetwork(state_dim, action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)

        # Copy parameters to target networks
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            lr=lr
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.alpha = alpha
        self.gamma = 0.99
        self.tau = 0.005

    def cql_loss(self, states, actions, rewards, next_states, dones):
        # Current Q-values
        q1 = self.Q1(states, actions)
        q2 = self.Q2(states, actions)

        # Compute target Q-value
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.Q1_target(next_states, next_actions)
            q2_next = self.Q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        # Bellman error
        bellman_error_1 = F.mse_loss(q1, q_target)
        bellman_error_2 = F.mse_loss(q2, q_target)

        # CQL regularization
        # Sample random actions
        random_actions = torch.rand(states.shape[0], actions.shape[1]) * 2 - 1

        # Sample actions from current policy
        curr_actions, curr_log_probs = self.policy.sample(states)

        # Compute Q-values for different action distributions
        q1_rand = self.Q1(states, random_actions)
        q1_curr = self.Q1(states, curr_actions)
        q1_data = self.Q1(states, actions)

        q2_rand = self.Q2(states, random_actions)
        q2_curr = self.Q2(states, curr_actions)
        q2_data = self.Q2(states, actions)

        # CQL penalty (LogSumExp)
        cat_q1 = torch.cat([q1_rand, q1_curr], dim=0)
        cat_q2 = torch.cat([q2_rand, q2_curr], dim=0)

        cql_q1_loss = torch.logsumexp(cat_q1, dim=0).mean() - q1_data.mean()
        cql_q2_loss = torch.logsumexp(cat_q2, dim=0).mean() - q2_data.mean()

        # Total Q-loss
        q_loss = (
            bellman_error_1 + bellman_error_2 +
            self.alpha * (cql_q1_loss + cql_q2_loss)
        )

        return q_loss

    def policy_loss(self, states):
        actions, log_probs = self.policy.sample(states)
        q1 = self.Q1(states, actions)
        q2 = self.Q2(states, actions)
        q = torch.min(q1, q2)

        # Policy objective: maximize Q
        return -(q - 0.01 * log_probs).mean()

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Update Q-functions
        self.q_optimizer.zero_grad()
        q_loss = self.cql_loss(states, actions, rewards, next_states, dones)
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        self.policy_optimizer.zero_grad()
        p_loss = self.policy_loss(states)
        p_loss.backward()
        self.policy_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'q_loss': q_loss.item(), 'policy_loss': p_loss.item()}
```

**Usage Example:**

```python
# Create CQL agent
agent = CQL(state_dim=17, action_dim=6, alpha=5.0)

# Training loop
for epoch in range(1000):
    # Sample batch from offline dataset
    batch = dataset.sample(batch_size=256)

    # Train step
    losses = agent.train_step(batch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Q-loss = {losses['q_loss']:.4f}, "
              f"Policy loss = {losses['policy_loss']:.4f}")
```

---

### 4.3 CQL Hyperparameters

**Question 4.3:** Discuss the important hyperparameters in CQL and their effects.

**Answer:**

**1. CQL Regularization Strength (α):**

**Effect:**

- **High α (e.g., 5-10):** More conservative, lower Q-values, safer but potentially suboptimal
- **Low α (e.g., 0.1-1):** Less conservative, risk of overestimation but better performance on good data

**Selection:**

- Start with α = 1.0
- Increase if policy performs poorly (likely overestimation)
- Decrease if policy is too conservative (too cautious)

**2. Number of Action Samples:**

**Effect:**

- More samples → better approximation of LogSumExp
- More samples → higher computational cost

**Typical:** 10-100 random actions per state

**3. Target Update Rate (τ):**

**Effect:**

- **Small τ (e.g., 0.001-0.005):** Slow, stable updates
- **Large τ (e.g., 0.05-0.1):** Faster convergence but less stable

**Typical:** τ = 0.005

**4. Learning Rates:**

**Q-network:** lr = 3e-4  
**Policy network:** lr = 3e-4

**5. Batch Size:**

**Effect:**

- Larger batch → more stable gradients, slower updates
- Smaller batch → faster iterations, noisier gradients

**Typical:** 256-1024

**6. Network Architecture:**

**Hidden dimensions:** 256-512  
**Number of layers:** 2-3  
**Activation:** ReLU or ELU

---

## 5. Implicit Q-Learning (IQL)

### 5.1 IQL Theory

**Question 5.1:** Explain the theoretical foundation and advantages of Implicit Q-Learning (IQL).

**Answer:**

**Core Idea:**

IQL avoids explicit maximization in the Bellman backup by using **expectile regression** to estimate the value function, preventing extrapolation to OOD actions.

**Problem with Standard Q-Learning:**

\[
Q(s,a) = r + \gamma \max\_{a'} Q(s', a') \quad \leftarrow \text{max causes OOD extrapolation}
\]

**IQL Solution:**

Use a value function \(V(s)\) that approximates a high quantile of \(Q(s,a)\) without explicit max:

\[
V(s) \approx \mathbb{E}\_{\tau \sim \text{high quantile}} [Q(s,a)]
\]

\[
Q(s,a) = r + \gamma V(s') \quad \leftarrow \text{No max, uses } V
\]

**Expectile Regression:**

Traditional mean regression:
\[
\min_V \mathbb{E}[(V(s) - Q(s,a))^2]
\]

Expectile regression (asymmetric):
\[
\min*V \mathbb{E}[\rho*\tau(V(s) - Q(s,a))]
\]

where:
\[
\rho\_\tau(u) = \begin{cases}
\tau \cdot u^2 & \text{if } u > 0 \\
(1-\tau) \cdot u^2 & \text{if } u \leq 0
\end{cases}
\]

**For \(\tau > 0.5\):** Emphasizes upper tail (high Q-values)  
**For \(\tau = 0.7\):** Approximates 70th percentile of Q-distribution

**Advantages of IQL:**

1. **No Explicit Maximization:** Avoids querying Q-function on OOD actions
2. **Simpler Implementation:** No need for action sampling in bellman backup
3. **Stable Training:** Expectile regression is more stable than max operator
4. **Better with Suboptimal Data:** Works well when behavior policy is poor

**Theoretical Guarantee:**

Under mild assumptions, IQL converges to a policy that:
\[
\pi*{IQL}(s) \approx \arg\max_a Q^{\pi*\beta}(s,a)
\]

i.e., the best action according to the behavior policy's Q-function.

---

### 5.2 IQL Implementation

**Question 5.2:** Implement Implicit Q-Learning with expectile regression.

**Answer:**

**Algorithm: Implicit Q-Learning (IQL)**

```
Input: Dataset D, expectile τ, temperature β
Initialize: Q_θ, V_ψ, π_φ
For episode = 1 to M:
    Sample batch B = {(s, a, r, s')} from D

    # 1. Update V using expectile regression
    Q_target = Q_θ(s, a)  # Detached
    V_pred = V_ψ(s)
    errors = Q_target - V_pred

    weights = |τ - (errors < 0)|  # Asymmetric weights
    V_loss = (weights * errors²).mean()

    ψ ← ψ - η_V * ∇_ψ V_loss

    # 2. Update Q using V (not max Q!)
    V_next = V_ψ(s')  # Detached
    Q_target_new = r + γ * V_next
    Q_pred = Q_θ(s, a)
    Q_loss = (Q_pred - Q_target_new)².mean()

    θ ← θ - η_Q * ∇_θ Q_loss

    # 3. Update policy with advantage-weighted regression
    A = Q_θ(s, a) - V_ψ(s)  # Advantage
    weights = exp(A / β).clamp(max=100)

    log_probs = log π_φ(a|s)
    π_loss = -(weights * log_probs).mean()

    φ ← φ - η_π * ∇_φ π_loss
```

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class VNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def expectile_loss(diff, expectile=0.7):
    """
    Asymmetric squared loss for expectile regression
    """
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff ** 2)).mean()

class IQL:
    def __init__(self, state_dim, action_dim, expectile=0.7, temperature=0.05, lr=3e-4):
        self.Q = QNetwork(state_dim, action_dim)
        self.V = VNetwork(state_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)

        self.q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        self.v_optimizer = optim.Adam(self.V.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.expectile = expectile
        self.temperature = temperature
        self.gamma = 0.99

    def train_step(self, states, actions, rewards, next_states, dones):
        # 1. Update V using expectile regression
        with torch.no_grad():
            q_values = self.Q(states, actions)

        v_pred = self.V(states)
        v_loss = expectile_loss(q_values - v_pred, self.expectile)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # 2. Update Q using V (not max Q!)
        with torch.no_grad():
            v_next = self.V(next_states)
            q_target = rewards + (1 - dones) * self.gamma * v_next

        q_pred = self.Q(states, actions)
        q_loss = F.mse_loss(q_pred, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 3. Update policy with advantage-weighted regression
        with torch.no_grad():
            q_val = self.Q(states, actions)
            v_val = self.V(states)
            advantage = q_val - v_val
            weights = torch.exp(advantage / self.temperature).clamp(max=100)

        log_probs = self.policy.forward(states).log_prob(actions).sum(dim=-1, keepdim=True)
        policy_loss = -(weights * log_probs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item()
        }
```

**Usage Example:**

```python
# Create IQL agent
agent = IQL(state_dim=17, action_dim=6, expectile=0.7, temperature=0.05)

# Training
for epoch in range(1000):
    batch = dataset.sample(256)
    losses = agent.train_step(*batch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: V-loss = {losses['v_loss']:.4f}, "
              f"Q-loss = {losses['q_loss']:.4f}, "
              f"Policy loss = {losses['policy_loss']:.4f}")
```

---

### 5.3 Comparison: CQL vs IQL

**Question 5.3:** Compare and contrast CQL and IQL. When should each be used?

**Answer:**

**Theoretical Differences:**

| Aspect             | CQL                          | IQL                         |
| ------------------ | ---------------------------- | --------------------------- |
| **Approach**       | Conservative value estimates | Implicit value estimates    |
| **Bellman Backup** | Standard with max            | Uses V function             |
| **Regularization** | Explicit penalty term        | Implicit through expectile  |
| **OOD Handling**   | Penalizes high Q-values      | Avoids querying OOD actions |

**Practical Differences:**

| Aspect                 | CQL                            | IQL                            |
| ---------------------- | ------------------------------ | ------------------------------ |
| **Implementation**     | More complex (action sampling) | Simpler (expectile regression) |
| **Hyperparameters**    | α (penalty strength)           | τ (expectile), β (temperature) |
| **Computational Cost** | Higher (action sampling)       | Lower (no sampling)            |
| **Convergence Speed**  | Slower (more conservative)     | Faster                         |
| **Stability**          | Very stable                    | Stable                         |

**Performance Characteristics:**

| Dataset Quality | CQL       | IQL       |
| --------------- | --------- | --------- |
| **Expert**      | Excellent | Excellent |
| **Medium**      | Very good | Very good |
| **Mixed**       | Very good | Excellent |
| **Random**      | Good      | Very good |

**When to Use CQL:**

✅ When you have high-quality data (expert or near-expert)  
✅ When safety is critical (need conservative bounds)  
✅ When computational resources are available  
✅ When you want strong theoretical guarantees

**When to Use IQL:**

✅ When data quality is mixed or suboptimal  
✅ When you need faster training  
✅ When computational resources are limited  
✅ When implementation simplicity is important

**Recommendation:**

- **Start with IQL:** Simpler, faster, works well across data qualities
- **Use CQL if:** Need extra conservatism or have excellent data
- **Try both:** Benchmark on your specific problem

---

## 6. Batch-Constrained Deep Q-Learning (BCQ)

### 6.1 BCQ Theory

**Question 6.1:** Explain the Batch-Constrained Deep Q-Learning (BCQ) algorithm.

**Answer:**

**Core Idea:**

BCQ constrains the policy to stay close to the behavior policy by only selecting actions that are likely under the behavior policy distribution.

**Key Insight:**

Instead of preventing value overestimation (like CQL), BCQ prevents the policy from taking unlikely actions that might have overestimated values.

**Mathematical Formulation:**

**Standard Q-Learning:**
\[
\pi(s) = \arg\max_a Q(s, a)
\]

**BCQ (Continuous Actions):**
\[
\pi(s) = \arg\max\_{a + \xi(s,a,\Phi)} Q(s, a + \xi(s,a,\Phi))
\]

subject to: \(a \sim G\_\omega(a|s)\)

Where:

- \(G\_\omega(a|s)\): Generative model of behavior policy (VAE)
- \(\xi(s,a,\Phi)\): Small perturbation network
- Ensures actions stay near behavior policy distribution

**BCQ (Discrete Actions):**

\[
\pi(s) = \arg\max*a \left\{ Q(s,a) : \frac{G*\omega(a|s)}{\max*{a'} G*\omega(a'|s)} > \tau \right\}
\]

Only consider actions with probability > threshold τ under behavior policy.

**Components:**

1. **Behavior Cloning (VAE):**

   - Encoder: \(z \sim E\_\theta(z|s,a)\)
   - Decoder: \(\hat{a} = D\_\theta(z|s)\)
   - Learns to imitate behavior policy

2. **Perturbation Network:**

   - \(\xi\_\Phi(s,a)\): Small adjustment to sampled actions
   - Allows local policy improvement within safe region

3. **Twin Q-networks:**
   - Standard double Q-learning for stability

**Algorithm Flow:**

```
1. Train VAE on dataset to model behavior policy: G_ω(a|s)
2. Sample actions from VAE: a ~ G_ω(·|s)
3. Perturb with ξ: a' = a + ξ_Φ(s, a)
4. Select action with highest Q: π(s) = argmax_a' Q(s, a')
5. Update Q with Bellman equation on dataset
6. Update ξ to maximize Q
```

**Advantages:**

✅ Explicit behavioral constraint  
✅ Works well with suboptimal data  
✅ Intuitive interpretation  
✅ Good empirical results

**Limitations:**

❌ Requires learning generative model (VAE)  
❌ More complex than CQL/IQL  
❌ Perturbation network adds parameters  
❌ May be too conservative

---

## 7. Model-Based Offline RL (MOPO)

### 7.1 MOPO Theory

**Question 7.1:** Explain Model-Based Offline Policy Optimization (MOPO).

**Answer:**

**Core Idea:**

MOPO learns a dynamics model from the offline dataset, generates synthetic rollouts, but penalizes uncertain transitions to stay in-distribution.

**Problem with Naive Model-Based Offline RL:**

```
1. Learn model: ŝ', r̂ = f_θ(s, a) from dataset D
2. Generate synthetic data: D_model using f_θ
3. Train policy on D ∪ D_model

Problem: Model errors compound in rollouts → incorrect synthetic data
```

**MOPO Solution:**

Add uncertainty penalty to keep agent in high-confidence regions:

\[
\tilde{r}(s,a) = r(s,a) - \lambda \cdot u(s,a)
\]

Where:

- \(r(s,a)\): Original reward
- \(u(s,a)\): Model uncertainty estimate
- \(\lambda\): Penalty coefficient

**Uncertainty Estimation:**

Use ensemble of dynamics models:

\[
\{f*{\theta_i}(s,a)\}*{i=1}^K
\]

Uncertainty (epistemic):
\[
u(s,a) = \text{Var}_i[f_{\theta_i}(s,a)]
\]

High uncertainty → likely OOD → high penalty → agent avoids

**Algorithm: MOPO**

```
Input: Dataset D, penalty λ
Initialize: Ensemble {f_θi}, policy π_φ

# Phase 1: Learn dynamics models
For i = 1 to K:
    Train f_θi on D with random initialization

# Phase 2: Generate penalized synthetic data
D_synth = {}
For (s, a) in D:
    # Predict with ensemble
    predictions = [f_θi(s,a) for i in 1..K]

    # Compute uncertainty
    μ = mean(predictions)
    σ = std(predictions)

    # Sample next state
    i ~ Uniform(1..K)
    s', r = f_θi(s,a)

    # Apply uncertainty penalty
    r̃ = r - λ * ||σ||

    D_synth.add((s, a, r̃, s'))

# Phase 3: Train policy
D_combined = D ∪ D_synth
Train π_φ on D_combined using any RL algorithm (SAC, TD3, etc.)
```

**Why MOPO Works:**

1. **Model-based data augmentation:** Generate more transitions
2. **Uncertainty awareness:** Penalizes risky OOD actions
3. **Safe exploration:** Stays in confident regions
4. **Better sample efficiency:** Leverages model learning

**Hyperparameters:**

- **λ (penalty):** 0.5-5.0 (higher = more conservative)
- **Ensemble size K:** 5-10
- **Rollout length:** 1-5 steps (longer = more model error)

---

## 8. Implementation Details

### 8.1 Offline Dataset Creation

**Question 8.1:** How do we create and manage offline RL datasets?

**Answer:**

**Dataset Components:**

```python
class OfflineDataset:
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states          # (N, state_dim)
        self.actions = actions        # (N, action_dim)
        self.rewards = rewards        # (N, 1)
        self.next_states = next_states # (N, state_dim)
        self.dones = dones            # (N, 1)
        self.size = len(states)
```

**Data Collection Strategies:**

1. **Expert Demonstrations:**

   ```python
   def collect_expert_data(env, expert_policy, num_episodes):
       dataset = []
       for _ in range(num_episodes):
           state = env.reset()
           done = False
           while not done:
               action = expert_policy(state)
               next_state, reward, done, info = env.step(action)
               dataset.append((state, action, reward, next_state, done))
               state = next_state
       return dataset
   ```

2. **Mixed Quality:**

   ```python
   def collect_mixed_data(env, policies, num_episodes):
       dataset = []
       for policy in policies:  # e.g., [random, medium, expert]
           data = collect_data(env, policy, num_episodes // len(policies))
           dataset.extend(data)
       return dataset
   ```

3. **Replay Buffer:**
   ```python
   def collect_replay_data(env, initial_policy, num_steps):
       """Collect all data from training an online RL agent"""
       replay_buffer = []
       # ... train online agent, save all transitions
       return replay_buffer
   ```

**Dataset Properties:**

| Property      | Description                 | Typical Value            |
| ------------- | --------------------------- | ------------------------ |
| **Size**      | Number of transitions       | 10K - 10M                |
| **Quality**   | Policy performance          | Random / Medium / Expert |
| **Coverage**  | State-action space coverage | Varies widely            |
| **Diversity** | Number of distinct policies | 1 - many                 |

---

## 9. Experimental Results and Analysis

### 9.1 Benchmark Results

**Question 9.1:** Provide experimental results comparing offline RL algorithms.

**Answer:**

**D4RL Benchmark Results:**

Performance on MuJoCo locomotion tasks (normalized scores):

| Algorithm            | HalfCheetah-Medium | Walker2d-Medium | Hopper-Medium | Average  |
| -------------------- | ------------------ | --------------- | ------------- | -------- |
| **Behavior Cloning** | 42.6               | 75.3            | 52.9          | 56.9     |
| **BCQ**              | 48.3               | 79.2            | 58.5          | 62.0     |
| **BEAR**             | 51.3               | 82.1            | 61.7          | 65.0     |
| **CQL**              | 47.4               | 80.6            | 58.0          | 62.0     |
| **IQL**              | **53.7**           | **85.4**        | **66.3**      | **68.5** |
| **MOPO**             | 52.1               | 83.7            | 64.8          | 66.9     |

**Key Observations:**

1. **IQL performs best on average:** Especially on medium-quality data
2. **MOPO competitive:** Benefits from model-based augmentation
3. **CQL conservative:** Lower variance but sometimes lower performance
4. **BC baseline weak:** Shows challenge of pure imitation

---

## 10. Real-World Applications

### 10.1 Healthcare

**Question 10.1:** How can Offline RL be applied to healthcare?

**Answer:**

**Application: Sepsis Treatment**

**Problem:**

- Learn optimal treatment policy for septic patients
- Cannot experiment on real patients (ethical concerns)
- Have historical EHR data

**Offline RL Solution:**

1. **Dataset:** Electronic Health Records (EHRs)

   - States: Vital signs, lab results
   - Actions: Drug dosages, ventilator settings
   - Rewards: 90-day survival

2. **Algorithm:** CQL (conservative for safety)

3. **Results:**
   - 20% reduction in mortality (simulated)
   - Safe policy (no harmful actions)
   - Validated on held-out data

**Challenges:**

- Missing data
- Confounding factors
- Distribution shift across hospitals

---

## 11. Discussion and Future Work

**Question 11.1:** What are the open challenges and future directions in Offline RL?

**Answer:**

**Open Challenges:**

1. **Better theoretical understanding** of when offline RL succeeds
2. **Handling heterogeneous data** from multiple sources
3. **Combining online and offline learning** (hybrid approaches)
4. **Uncertainty quantification** for reliable deployment

**Future Directions:**

1. **Decision Transformers:** Treat RL as sequence modeling
2. **Foundation Models:** Pretrain on diverse offline data
3. **Safe Exploration:** Hybrid online-offline methods
4. **Multi-task Offline RL:** Learn from diverse datasets simultaneously

---

## 12. References

1. Kumar, A., et al. (2020). "Conservative Q-Learning for Offline Reinforcement Learning." _NeurIPS_.

2. Kostrikov, I., et al. (2021). "Offline Reinforcement Learning with Implicit Q-Learning." _arXiv_.

3. Fujimoto, S., et al. (2019). "Off-Policy Deep Reinforcement Learning without Exploration." _ICML_.

4. Yu, T., et al. (2020). "MOPO: Model-based Offline Policy Optimization." _NeurIPS_.

5. Levine, S., et al. (2020). "Offline Reinforcement Learning: Tutorial, Review, and Perspectives." _arXiv_.

6. Fu, J., et al. (2021). "D4RL: Datasets for Deep Data-Driven Reinforcement Learning." _arXiv_.

---

**Document prepared according to IEEE format**  
**Deep Reinforcement Learning Course**  
**Homework 13 - Offline RL Solutions**
