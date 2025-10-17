# HW8: Exploration Methods in Reinforcement Learning - Complete Solutions

**Course:** Deep Reinforcement Learning  
**Assignment:** HW8 - Exploration Methods  
**Format:** IEEE  
**Author:** [Student Name]  
**Date:** 2024

---

## Abstract

This document presents comprehensive solutions to HW8 on Exploration Methods in Reinforcement Learning. We address the fundamental exploration-exploitation dilemma through theoretical analysis and practical implementation of various exploration strategies including multi-armed bandits, Upper Confidence Bound (UCB), Thompson Sampling, count-based methods, and intrinsic motivation approaches. Each problem is solved with detailed mathematical derivations, algorithmic implementations, and empirical analysis following IEEE formatting standards.

**Keywords:** Reinforcement Learning, Exploration, Multi-Armed Bandits, UCB, Thompson Sampling, Intrinsic Motivation, RND

---

## Table of Contents

1. [Multi-Armed Bandits Fundamentals](#section-1)
2. [Epsilon-Greedy vs UCB Analysis](#section-2)
3. [Thompson Sampling Implementation](#section-3)
4. [Count-Based Exploration](#section-4)
5. [Intrinsic Motivation Methods](#section-5)
6. [Random Network Distillation (RND)](#section-6)
7. [Noisy Networks for Exploration](#section-7)
8. [Bootstrap DQN](#section-8)
9. [Regret Analysis](#section-9)
10. [Empirical Comparisons](#section-10)

---

<a name="section-1"></a>

## 1. Multi-Armed Bandits Fundamentals

### Problem 1.1: Understanding the MAB Framework

**Question:** Define the Multi-Armed Bandit problem formally. What is regret and why is it important?

**Solution:**

#### A. Formal Definition

The Multi-Armed Bandit (MAB) problem is defined by a tuple $(K, \mathcal{R})$ where:

- $K$: Number of arms (actions)
- $\mathcal{R} = \{R_1, R_2, ..., R_K\}$: Set of reward distributions

Each arm $i \in \{1, ..., K\}$ has an associated reward distribution with:

```
r_i ~ R_i, with mean μ_i = E[r_i]
```

At each time step $t$:

1. Agent selects arm $a_t \in \{1, ..., K\}$
2. Environment returns reward $r_t \sim R_{a_t}$
3. Agent updates strategy

**Objective:** Maximize cumulative reward over $T$ time steps:

```
max E[∑_{t=1}^T r_t]
```

#### B. Regret Definition

**Definition:** Regret measures the difference between the expected cumulative reward of the optimal policy and the agent's policy.

Formally, the cumulative regret after $T$ steps is:

$$R(T) = T \cdot \mu^* - \mathbb{E}\left[\sum_{t=1}^T r_t\right]$$

where $\mu^* = \max_{i} \mu_i$ is the expected reward of the best arm.

**Alternative formulation:**

$$R(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})$$

This can also be expressed in terms of suboptimality gaps:

$$R(T) = \sum_{i=1}^K \Delta_i \mathbb{E}[N_i(T)]$$

where:

- $\Delta_i = \mu^* - \mu_i$ is the suboptimality gap for arm $i$
- $N_i(T)$ is the number of times arm $i$ has been pulled by time $T$

#### C. Importance of Regret

1. **Performance Metric:** Directly measures the cost of learning
2. **Algorithm Comparison:** Enables principled comparison of exploration strategies
3. **Theoretical Guarantees:** Algorithms can be analyzed via regret bounds
4. **Optimal Trade-off:** Good algorithms achieve sublinear regret $R(T) = o(T)$

**Key Insight:** Sublinear regret implies that average regret per step → 0 as T → ∞, meaning the algorithm eventually learns to exploit optimally.

### Problem 1.2: Lower Bounds on Regret

**Question:** Prove that for any algorithm, there exists a MAB instance where the expected regret is $\Omega(\sqrt{KT})$.

**Solution:**

This is a fundamental result from information theory. We prove using Pinsker's inequality and considering adversarial problem instances.

#### Proof Sketch:

**1. Setup:**
Consider two arms with Bernoulli reward distributions:

- Arm 1: $p_1 = 0.5 + \epsilon$
- Arm 2: $p_2 = 0.5 - \epsilon$

where $\epsilon > 0$ is small.

**2. Information-Theoretic Argument:**

Any algorithm must distinguish between two hypotheses:

- $H_0$: Arm 1 is better
- $H_1$: Arm 2 is better

The KL divergence between observations from the two arms is:
$$D_{KL}(P_1 || P_2) = O(\epsilon^2)$$

**3. Sample Complexity:**

To distinguish with high probability, need $\Omega(1/\epsilon^2)$ samples.

**4. Regret Calculation:**

If $T < K/\epsilon^2$, algorithm cannot identify optimal arm with high probability, leading to expected regret:
$$R(T) = \Omega(K\epsilon \cdot T) = \Omega(\sqrt{KT})$$

when setting $\epsilon = \sqrt{K/T}$.

**Theorem (Lai & Robbins, 1985):**
For any consistent algorithm (one that identifies optimal arm eventually):

$$\liminf_{T \to \infty} \frac{R(T)}{\log T} \geq \sum_{i: \Delta_i > 0} \frac{\Delta_i}{D_{KL}(\mu_i || \mu^*)}$$

This shows logarithmic regret is optimal in the asymptotic regime.

---

<a name="section-2"></a>

## 2. Epsilon-Greedy vs UCB Analysis

### Problem 2.1: Epsilon-Greedy Algorithm

**Question:** Implement epsilon-greedy and analyze its regret. Why does it have linear regret?

**Solution:**

#### A. Algorithm Implementation

```python
import numpy as np

class EpsilonGreedy:
    """
    Epsilon-Greedy Multi-Armed Bandit Algorithm
    """
    def __init__(self, num_arms, epsilon=0.1):
        """
        Initialize epsilon-greedy algorithm

        Args:
            num_arms (int): Number of arms
            epsilon (float): Exploration probability
        """
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = np.zeros(num_arms)  # N_i(t)
        self.values = np.zeros(num_arms)  # Q̂_i(t)

    def select_action(self):
        """
        Select action using epsilon-greedy policy

        Returns:
            int: Selected arm index
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.num_arms)
        else:
            # Exploit: greedy action
            return np.argmax(self.values)

    def update(self, arm, reward):
        """
        Update value estimates using incremental mean

        Args:
            arm (int): Selected arm
            reward (float): Observed reward
        """
        self.counts[arm] += 1
        n = self.counts[arm]

        # Incremental update: Q_n = Q_{n-1} + (1/n)(r_n - Q_{n-1})
        self.values[arm] += (reward - self.values[arm]) / n

    def run_episode(self, env, num_steps):
        """
        Run complete episode

        Args:
            env: Bandit environment
            num_steps (int): Number of steps

        Returns:
            tuple: (rewards, regrets, optimal_actions)
        """
        rewards = []
        regrets = []
        optimal_actions = []

        for t in range(num_steps):
            # Select and execute action
            arm = self.select_action()
            reward = env.pull(arm)

            # Update estimates
            self.update(arm, reward)

            # Record metrics
            rewards.append(reward)
            regrets.append(env.optimal_reward - reward)
            optimal_actions.append(arm == env.optimal_arm)

        return np.array(rewards), np.array(regrets), np.array(optimal_actions)
```

#### B. Regret Analysis

**Theorem:** Fixed epsilon-greedy has expected regret:

$$\mathbb{E}[R(T)] = \Omega(T)$$

**Proof:**

At each step $t$:

- With probability $\epsilon$: Explore randomly
- With probability $1-\epsilon$: Exploit greedy choice

Expected regret at step $t$:
$$\mathbb{E}[r_t^* - r_t] = \epsilon \cdot \mathbb{E}[\text{random regret}] + (1-\epsilon) \cdot \mathbb{E}[\text{greedy regret}]$$

Even after converging to correct value estimates:
$$\mathbb{E}[r_t^* - r_t] = \epsilon \cdot \frac{1}{K}\sum_{i=1}^K \Delta_i > 0$$

Therefore:
$$R(T) = \sum_{t=1}^T \mathbb{E}[r_t^* - r_t] \geq c \epsilon T$$

for some constant $c > 0$, giving linear regret $\Theta(T)$.

#### C. Decaying Epsilon

To achieve sublinear regret, use time-dependent $\epsilon_t$:

$$\epsilon_t = \min\left(1, \frac{c K}{d^2 t}\right)$$

where $d = \min_i \Delta_i$ is the minimum gap.

**Result:** This achieves regret bound:
$$R(T) = O\left(\frac{K \log T}{d}\right)$$

```python
class DecayingEpsilonGreedy(EpsilonGreedy):
    """Epsilon-greedy with decaying exploration"""

    def __init__(self, num_arms, c=1.0, d=0.1):
        super().__init__(num_arms, epsilon=1.0)
        self.c = c
        self.d = d
        self.t = 0

    def select_action(self):
        self.t += 1
        # Decay epsilon over time
        self.epsilon = min(1, self.c * self.num_arms / (self.d**2 * self.t))
        return super().select_action()
```

### Problem 2.2: Upper Confidence Bound (UCB)

**Question:** Implement UCB1 algorithm and prove its logarithmic regret bound.

**Solution:**

#### A. UCB1 Algorithm

The UCB1 algorithm selects the arm with highest upper confidence bound:

$$a_t = \argmax_{i} \left[ \hat{\mu}_i + \sqrt{\frac{2\log t}{N_i(t)}} \right]$$

where:

- $\hat{\mu}_i$: Empirical mean reward of arm $i$
- $N_i(t)$: Number of times arm $i$ pulled
- $t$: Total number of steps

**Implementation:**

```python
class UCB1:
    """
    Upper Confidence Bound Algorithm
    """
    def __init__(self, num_arms, c=np.sqrt(2)):
        """
        Args:
            num_arms (int): Number of arms
            c (float): Exploration constant (typically √2)
        """
        self.num_arms = num_arms
        self.c = c
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.t = 0

    def select_action(self):
        """
        Select action using UCB criterion

        Returns:
            int: Selected arm
        """
        self.t += 1

        # Initially, try all arms once
        if self.t <= self.num_arms:
            return self.t - 1

        # Compute UCB values
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.t) / (self.counts + 1e-10)
        )

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Update estimates"""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def get_confidence_bounds(self):
        """
        Get current confidence bounds

        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        if self.t == 0:
            return np.zeros(self.num_arms), np.zeros(self.num_arms)

        confidence_width = self.c * np.sqrt(
            np.log(self.t) / (self.counts + 1e-10)
        )

        return (self.values - confidence_width,
                self.values + confidence_width)
```

#### B. Regret Bound Proof

**Theorem (Auer et al., 2002):** For UCB1:

$$\mathbb{E}[R(T)] \leq 8\sum_{i: \Delta_i > 0} \frac{\log T}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right)\sum_{i=1}^K \Delta_i$$

**Proof Sketch:**

**1. Key Insight:** An suboptimal arm $i$ is selected at time $t$ only if:
$$\hat{\mu}_i + \sqrt{\frac{2\log t}{N_i(t)}} \geq \mu^*$$

**2. Decompose Events:** This happens if either:

- (A) $\hat{\mu}_i$ overestimates: $\hat{\mu}_i \geq \mu_i + \sqrt{\frac{2\log t}{N_i(t)}}$
- (B) $\hat{\mu}^*$ underestimates: $\hat{\mu}^* \leq \mu^* - \sqrt{\frac{2\log t}{N^*(t)}}$
- (C) $2\sqrt{\frac{2\log t}{N_i(t)}} \geq \Delta_i$

**3. Bound Probabilities:**

Using Hoeffding's inequality:
$$P(\hat{\mu}_i \geq \mu_i + \sqrt{\frac{2\log t}{N_i(t)}}) \leq \frac{1}{t^4}$$

**4. Expected Pulls:** Event (C) happens at most:
$$N_i(t) \leq \frac{8\log T}{\Delta_i^2}$$

Events (A) and (B) happen with total probability $O(1/t^2)$, summing to $O(1)$ over all $T$.

**5. Total Regret:**
$$\mathbb{E}[R(T)] = \sum_{i: \Delta_i > 0} \Delta_i \mathbb{E}[N_i(T)] \leq \sum_{i: \Delta_i > 0} \frac{8\log T}{\Delta_i} + O(1)$$

This gives **logarithmic regret** $O(\log T)$, exponentially better than epsilon-greedy's linear regret!

---

<a name="section-3"></a>

## 3. Thompson Sampling Implementation

### Problem 3.1: Bayesian Approach to Bandits

**Question:** Implement Thompson Sampling for Bernoulli bandits and analyze its performance.

**Solution:**

#### A. Theoretical Foundation

Thompson Sampling is a Bayesian approach that maintains posterior distribution over each arm's reward parameter.

**For Bernoulli rewards:**

- Prior: $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$
- Likelihood: $r \sim \text{Bernoulli}(\theta_i)$
- Posterior: $\theta_i | r \sim \text{Beta}(\alpha_i + r, \beta_i + (1-r))$

**Algorithm:**

1. Sample $\tilde{\theta}_i \sim \text{Beta}(\alpha_i, \beta_i)$ for each arm $i$
2. Select $a_t = \argmax_i \tilde{\theta}_i$
3. Observe reward $r_t$
4. Update: $\alpha_{a_t} \leftarrow \alpha_{a_t} + r_t$, $\beta_{a_t} \leftarrow \beta_{a_t} + (1 - r_t)$

#### B. Implementation

```python
class ThompsonSampling:
    """
    Thompson Sampling for Bernoulli Multi-Armed Bandits
    """
    def __init__(self, num_arms, alpha_prior=1.0, beta_prior=1.0):
        """
        Args:
            num_arms (int): Number of arms
            alpha_prior (float): Prior alpha parameter (pseudo-successes)
            beta_prior (float): Prior beta parameter (pseudo-failures)
        """
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms) * alpha_prior
        self.beta = np.ones(num_arms) * beta_prior
        self.counts = np.zeros(num_arms)

    def select_action(self):
        """
        Sample from posterior and select action

        Returns:
            int: Selected arm based on Thompson sampling
        """
        # Sample from Beta posterior for each arm
        samples = np.random.beta(self.alpha, self.beta)

        # Select arm with highest sample
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update posterior distribution

        Args:
            arm (int): Selected arm
            reward (float): Observed reward (0 or 1 for Bernoulli)
        """
        self.counts[arm] += 1

        # Bayesian update
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)

    def get_posterior_stats(self):
        """
        Get posterior statistics for each arm

        Returns:
            dict: Mean, variance, and credible intervals
        """
        mean = self.alpha / (self.alpha + self.beta)
        var = (self.alpha * self.beta) / (
            (self.alpha + self.beta)**2 * (self.alpha + self.beta + 1)
        )

        # 95% credible intervals
        from scipy.stats import beta
        lower = beta.ppf(0.025, self.alpha, self.beta)
        upper = beta.ppf(0.975, self.alpha, self.beta)

        return {
            'mean': mean,
            'variance': var,
            'credible_interval': (lower, upper)
        }

    def probability_best_arm(self):
        """
        Compute probability each arm is optimal (via sampling)

        Returns:
            np.array: Probability each arm is best
        """
        num_samples = 10000
        samples = np.random.beta(
            self.alpha[:, None],
            self.beta[:, None],
            size=(self.num_arms, num_samples)
        )

        best_arm_samples = np.argmax(samples, axis=0)
        probabilities = np.array([
            np.mean(best_arm_samples == i) for i in range(self.num_arms)
        ])

        return probabilities
```

#### C. Gaussian Thompson Sampling

For continuous rewards with unknown mean:

```python
class GaussianThompsonSampling:
    """Thompson Sampling for Gaussian rewards"""

    def __init__(self, num_arms, prior_mean=0.0, prior_precision=1.0,
                 noise_precision=1.0):
        """
        Assumes known noise variance, unknown mean

        Args:
            num_arms (int): Number of arms
            prior_mean (float): Prior mean
            prior_precision (float): Prior precision (1/variance)
            noise_precision (float): Noise precision (1/σ²)
        """
        self.num_arms = num_arms
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision

        # Posterior parameters (Normal-Gamma conjugate)
        self.post_mean = np.ones(num_arms) * prior_mean
        self.post_precision = np.ones(num_arms) * prior_precision
        self.counts = np.zeros(num_arms)

    def select_action(self):
        """Sample from posterior Normal distribution"""
        post_std = 1.0 / np.sqrt(self.post_precision)
        samples = np.random.normal(self.post_mean, post_std)
        return np.argmax(samples)

    def update(self, arm, reward):
        """Bayesian update for Normal-Normal model"""
        self.counts[arm] += 1
        n = self.counts[arm]

        # Update posterior (conjugate update)
        precision = self.prior_precision + n * self.noise_precision
        mean = (self.prior_precision * self.prior_mean +
                self.noise_precision * reward * n) / precision

        self.post_mean[arm] = mean
        self.post_precision[arm] = precision
```

#### D. Regret Analysis

**Theorem (Agrawal & Goyal, 2012):** For Thompson Sampling on Bernoulli bandits:

$$\mathbb{E}[R(T)] = O\left(\sqrt{KT\log T}\right)$$

**Key Properties:**

1. **Information-Theoretically Optimal:** Matches lower bound in certain settings
2. **Problem-Dependent Bounds:** Can achieve $O(\log T)$ regret with proper priors
3. **Natural Exploration:** No tuning parameters needed
4. **Computational Efficiency:** Just sample and argmax

**Intuition:** Thompson Sampling naturally balances:

- **Exploration:** Uncertain arms have high posterior variance → higher chance of sampling
- **Exploitation:** Arms with high posterior mean → higher chance of sampling

---

<a name="section-4"></a>

## 4. Count-Based Exploration

### Problem 4.1: Exploration Bonuses

**Question:** Implement count-based exploration with $1/\sqrt{N}$ bonuses. Explain the theoretical motivation.

**Solution:**

#### A. Theoretical Foundation

**Motivation:** In MDPs, we want to encourage visiting rarely-seen states.

**Optimistic Initialization Principle:**

- Add bonus proportional to uncertainty
- Uncertainty decreases with visit counts
- Form: $\text{bonus} = \beta / \sqrt{N(s,a)}$

**Derivation from Hoeffding Bound:**

With $N(s,a)$ samples, the confidence interval for $Q(s,a)$ is:

$$|Q(s,a) - \hat{Q}(s,a)| \leq \sqrt{\frac{2\log(1/\delta)}{N(s,a)}}$$

Setting $\beta = \sqrt{2\log(1/\delta)}$ gives exploration bonus.

#### B. Implementation

```python
class CountBasedExploration:
    """
    Count-based exploration for tabular MDPs
    """
    def __init__(self, num_states, num_actions, beta=0.1, gamma=0.99):
        """
        Args:
            num_states (int): Number of states
            num_actions (int): Number of actions
            beta (float): Exploration bonus coefficient
            gamma (float): Discount factor
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.beta = beta
        self.gamma = gamma

        # Initialize Q-values and counts
        self.Q = np.zeros((num_states, num_actions))
        self.counts = np.zeros((num_states, num_actions))

    def exploration_bonus(self, state, action):
        """
        Compute exploration bonus

        Args:
            state (int): Current state
            action (int): Action

        Returns:
            float: Exploration bonus
        """
        n = self.counts[state, action]
        if n == 0:
            return self.beta  # Maximum bonus for unvisited
        return self.beta / np.sqrt(n)

    def select_action(self, state):
        """
        Select action using optimistic values

        Args:
            state (int): Current state

        Returns:
            int: Selected action
        """
        # Compute optimistic Q-values
        bonuses = np.array([
            self.exploration_bonus(state, a)
            for a in range(self.num_actions)
        ])

        optimistic_values = self.Q[state] + bonuses

        # Select greedy action w.r.t. optimistic values
        return np.argmax(optimistic_values)

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update with count

        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Observed reward
            next_state (int): Next state
            done (bool): Episode termination flag
        """
        self.counts[state, action] += 1

        # Learning rate: 1/N(s,a)
        alpha = 1.0 / self.counts[state, action]

        # TD target
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        # Q-learning update
        self.Q[state, action] += alpha * (target - self.Q[state, action])
```

#### C. R-Max Algorithm

**Idea:** Initialize all Q-values optimistically to $R_{max}$, the maximum possible reward.

```python
class RMax:
    """
    R-Max Algorithm for exploration
    """
    def __init__(self, num_states, num_actions, r_max=1.0,
                 gamma=0.99, m=5):
        """
        Args:
            num_states (int): Number of states
            num_actions (int): Number of actions
            r_max (float): Maximum reward
            gamma (float): Discount factor
            m (int): Threshold for 'known' state-action pairs
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.gamma = gamma
        self.m = m  # Visit threshold

        # Initialize optimistically
        self.Q = np.ones((num_states, num_actions)) * r_max / (1 - gamma)
        self.counts = np.zeros((num_states, num_actions))
        self.known = np.zeros((num_states, num_actions), dtype=bool)

        # Model for known states
        self.R = np.zeros((num_states, num_actions))
        self.T = np.zeros((num_states, num_actions, num_states))

    def is_known(self, state, action):
        """Check if state-action is sufficiently explored"""
        return self.counts[state, action] >= self.m

    def select_action(self, state):
        """Select greedy action w.r.t. current Q"""
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        """Update model and replan"""
        self.counts[state, action] += 1
        n = self.counts[state, action]

        # Update model incrementally
        self.R[state, action] += (reward - self.R[state, action]) / n
        self.T[state, action] += (
            (np.eye(self.num_states)[next_state] - self.T[state, action]) / n
        )

        # Mark as known if threshold reached
        if n >= self.m:
            self.known[state, action] = True

        # Recompute Q-values via value iteration on model
        self.value_iteration()

    def value_iteration(self, max_iters=100, tol=1e-4):
        """Compute Q-values via value iteration on learned model"""
        for _ in range(max_iters):
            Q_old = self.Q.copy()

            for s in range(self.num_states):
                for a in range(self.num_actions):
                    if self.known[s, a]:
                        # Use learned model
                        expected_value = np.sum(
                            self.T[s, a] * np.max(self.Q, axis=1)
                        )
                        self.Q[s, a] = self.R[s, a] + self.gamma * expected_value
                    # else: keep optimistic value

            if np.max(np.abs(Q_old - self.Q)) < tol:
                break
```

---

<a name="section-5"></a>

## 5. Intrinsic Motivation Methods

### Problem 5.1: Prediction Error Bonus

**Question:** Implement a forward model with prediction error as intrinsic reward. Discuss the "noisy TV" problem.

**Solution:**

#### A. Forward Model Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardModel(nn.Module):
    """
    Forward dynamics model: predicts next state from current state and action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension
            hidden_dim (int): Hidden layer size
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        """
        Predict next state

        Args:
            state: Current state tensor
            action: Action tensor (one-hot or continuous)

        Returns:
            torch.Tensor: Predicted next state
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class ForwardModelExploration:
    """
    Exploration via forward model prediction error
    """
    def __init__(self, state_dim, action_dim, eta=0.1, lr=1e-3):
        """
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension
            eta (float): Intrinsic reward scaling
            lr (float): Learning rate for forward model
        """
        self.model = ForwardModel(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.eta = eta

    def intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic reward based on prediction error

        Args:
            state: Current state
            action: Action taken
            next_state: Observed next state

        Returns:
            float: Intrinsic reward (prediction error)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # Predict next state
            pred_next_state = self.model(state_tensor, action_tensor)

            # Compute prediction error (MSE)
            error = F.mse_loss(pred_next_state, next_state_tensor)

            return self.eta * error.item()

    def update_model(self, state, action, next_state):
        """
        Update forward model via supervised learning

        Args:
            state: Current state
            action: Action taken
            next_state: Observed next state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Forward pass
        pred_next_state = self.model(state_tensor, action_tensor)

        # Compute loss
        loss = F.mse_loss(pred_next_state, next_state_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

#### B. The Noisy TV Problem

**Problem Statement:** Prediction error can fail when the environment contains:

1. **Stochastic dynamics** (inherently unpredictable)
2. **Irrelevant distractors** (e.g., TV showing random content)

**Example:**

- Agent in room with TV showing white noise
- TV state is unpredictable but irrelevant to task
- Forward model cannot predict TV pixels
- Agent gets stuck watching TV (high prediction error = high bonus)

**Why It Fails:**
$$r_{\text{intrinsic}} = \eta \cdot \mathbb{E}_{s,a}[||f(s,a) - s'||^2]$$

This bonus persists for aleatoric (irreducible) uncertainty, not just epistemic (learnable) uncertainty.

#### C. Solution: ICM (Intrinsic Curiosity Module)

**Key Idea:** Only be curious about features controllable by agent's actions.

**Architecture:**

1. **Feature Network:** $\phi: S \rightarrow \mathbb{R}^d$ (learned representations)
2. **Forward Model:** $\hat{\phi}(s') = f(\phi(s), a)$
3. **Inverse Model:** $\hat{a} = g(\phi(s), \phi(s'))$

**Intrinsic Reward:**
$$r_i = \eta \cdot ||\hat{\phi}(s') - \phi(s')||^2$$

computed in feature space, not raw state space.

```python
class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al., 2017)
    """
    def __init__(self, state_dim, action_dim, feature_dim=64, hidden_dim=256):
        super().__init__()

        # Feature encoder
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Inverse model: φ(s), φ(s') -> a
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Forward model: φ(s), a -> φ(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, state, action, next_state):
        """
        Compute intrinsic reward and losses

        Returns:
            dict: intrinsic_reward, forward_loss, inverse_loss
        """
        # Encode states
        phi_s = self.feature_net(state)
        phi_s_next = self.feature_net(next_state)

        # Forward model prediction
        phi_s_next_pred = self.forward_model(
            torch.cat([phi_s, action], dim=-1)
        )

        # Inverse model prediction
        action_pred = self.inverse_model(
            torch.cat([phi_s, phi_s_next], dim=-1)
        )

        # Losses
        forward_loss = 0.5 * F.mse_loss(phi_s_next_pred, phi_s_next.detach())
        inverse_loss = F.mse_loss(action_pred, action)

        # Intrinsic reward (forward prediction error in feature space)
        intrinsic_reward = 0.5 * ((phi_s_next_pred - phi_s_next.detach())**2).sum(dim=-1)

        return {
            'intrinsic_reward': intrinsic_reward.detach(),
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss
        }
```

**Training Objective:**
$$\mathcal{L} = \beta \mathcal{L}_{\text{forward}} + (1-\beta) \mathcal{L}_{\text{inverse}}$$

where typically $\beta = 0.2$.

**Advantages:**

- Filters out uncontrollable stochasticity
- Focuses on task-relevant features
- Solves noisy TV problem

---

<a name="section-6"></a>

## 6. Random Network Distillation (RND)

### Problem 6.1: Implement RND

**Question:** Implement Random Network Distillation and explain why it provides meaningful exploration bonuses.

**Solution:**

#### A. Theoretical Foundation

**Key Insight:** Prediction error of random features serves as novelty detector.

**Components:**

1. **Target Network** $f: S \rightarrow \mathbb{R}^k$ - randomly initialized, fixed
2. **Predictor Network** $\hat{f}: S \rightarrow \mathbb{R}^k$ - trained to match target

**Intrinsic Reward:**
$$r_i(s) = ||\hat{f}(s; \theta) - f(s)||^2$$

**Why It Works:**

- Novel states: predictor hasn't seen → high error → high bonus
- Familiar states: predictor trained → low error → low bonus
- Naturally handles stochasticity (both networks see same randomness)

#### B. Implementation

```python
class RNDModel(nn.Module):
    """
    Random Network Distillation for exploration
    """
    def __init__(self, state_dim, output_dim=64, hidden_dim=256):
        """
        Args:
            state_dim (int): State dimension
            output_dim (int): Feature dimension
            hidden_dim (int): Hidden layer size
        """
        super().__init__()

        # Target network (random, fixed)
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        """
        Compute target and prediction

        Args:
            state: State tensor

        Returns:
            tuple: (target_features, predicted_features)
        """
        target_feat = self.target(state)
        pred_feat = self.predictor(state)
        return target_feat, pred_feat

    def intrinsic_reward(self, state):
        """
        Compute intrinsic reward (prediction error)

        Args:
            state: State tensor

        Returns:
            torch.Tensor: Intrinsic rewards
        """
        target_feat, pred_feat = self.forward(state)
        return ((pred_feat - target_feat.detach())**2).mean(dim=-1)

class RNDExploration:
    """
    Complete RND exploration module with normalization
    """
    def __init__(self, state_dim, output_dim=64, lr=1e-4,
                 reward_scale=1.0, update_proportion=0.25):
        """
        Args:
            state_dim (int): State dimension
            output_dim (int): RND feature dimension
            lr (float): Learning rate for predictor
            reward_scale (float): Intrinsic reward scaling
            update_proportion (float): Proportion of data for updates
        """
        self.rnd = RNDModel(state_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=lr)
        self.reward_scale = reward_scale
        self.update_proportion = update_proportion

        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_momentum = 0.99

    def compute_intrinsic_reward(self, states, normalize=True):
        """
        Compute and optionally normalize intrinsic rewards

        Args:
            states: Batch of states
            normalize (bool): Whether to normalize rewards

        Returns:
            np.array: Intrinsic rewards
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states)
            rewards = self.rnd.intrinsic_reward(state_tensor).numpy()

        if normalize:
            # Update running statistics
            self.reward_mean = (
                self.reward_momentum * self.reward_mean +
                (1 - self.reward_momentum) * rewards.mean()
            )
            self.reward_std = (
                self.reward_momentum * self.reward_std +
                (1 - self.reward_momentum) * rewards.std()
            )

            # Normalize
            rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)

        return self.reward_scale * rewards

    def update(self, states):
        """
        Update predictor network

        Args:
            states: Batch of states

        Returns:
            float: Prediction loss
        """
        # Random subset for update (prevents overfitting)
        num_samples = int(len(states) * self.update_proportion)
        indices = np.random.choice(len(states), num_samples, replace=False)
        states_subset = states[indices]

        state_tensor = torch.FloatTensor(states_subset)

        # Forward pass
        target_feat, pred_feat = self.rnd(state_tensor)

        # MSE loss
        loss = F.mse_loss(pred_feat, target_feat.detach())

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

#### C. Integration with PPO

```python
def compute_gae_with_intrinsic_rewards(rewards_ext, rewards_int,
                                       values_ext, values_int,
                                       dones, gamma=0.99, gam_int=0.99,
                                       lam=0.95):
    """
    Compute Generalized Advantage Estimation with intrinsic rewards

    Args:
        rewards_ext: Extrinsic rewards
        rewards_int: Intrinsic rewards (from RND)
        values_ext: Extrinsic value estimates
        values_int: Intrinsic value estimates
        dones: Episode termination flags
        gamma: Discount factor for extrinsic
        gam_int: Discount factor for intrinsic (typically 0.99)
        lam: GAE parameter

    Returns:
        tuple: (advantages, returns_ext, returns_int)
    """
    advantages = []
    gae = 0

    T = len(rewards_ext)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value_ext = 0
            next_value_int = 0
        else:
            next_value_ext = values_ext[t + 1]
            next_value_int = values_int[t + 1]

        # TD errors
        delta_ext = rewards_ext[t] + gamma * next_value_ext * (1 - dones[t]) - values_ext[t]
        delta_int = rewards_int[t] + gam_int * next_value_int - values_int[t]

        # Combined delta
        delta = delta_ext + delta_int

        # GAE
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = np.array(advantages)
    returns_ext = advantages + values_ext
    returns_int = rewards_int + gam_int * np.roll(values_int, -1)

    return advantages, returns_ext, returns_int
```

#### D. Why RND Works

**1. Non-stationary Target:**

- Each state has unique random projection
- Novel states produce unpredictable features

**2. Handles Stochasticity:**

- Both networks see same random features
- Stochastic transitions don't cause spurious bonuses

**3. Exploration in Deep RL:**

- No need to count high-dimensional states
- Differentiable, scales to large spaces
- Works with function approximation

**Empirical Results (Burda et al., 2018):**

- Solves Montezuma's Revenge (hard exploration game)
- Discovers over 15 rooms without extrinsic reward
- Robust across different environments

---

<a name="section-7"></a>

## 7. Noisy Networks for Exploration

### Problem 7.1: Noisy Linear Layers

**Question:** Implement Noisy Networks and compare with epsilon-greedy exploration.

**Solution:**

#### A. Theoretical Foundation

**Key Idea:** Add learnable noise to network weights for state-dependent exploration.

**Noisy Linear Layer:**
$$y = (\mu^w + \sigma^w \odot \epsilon^w)x + \mu^b + \sigma^b \odot \epsilon^b$$

where:

- $\mu^w, \mu^b$: Learnable mean parameters
- $\sigma^w, \sigma^b$: Learnable standard deviations
- $\epsilon^w, \epsilon^b$: Random noise sampled from $\mathcal{N}(0,1)$

**Factorized Gaussian Noise (efficient):**
$$\epsilon^w_{ij} = f(\epsilon_i) \cdot f(\epsilon_j)$$

where $f(x) = \text{sgn}(x)\sqrt{|x|}$

#### B. Implementation

```python
class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer (Fortunato et al., 2018)
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        """
        Args:
            in_features (int): Input dimension
            out_features (int): Output dimension
            std_init (float): Initial standard deviation
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.register_buffer('weight_epsilon',
                           torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / np.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / np.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Factorized Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

class NoisyDQN(nn.Module):
    """
    DQN with Noisy Networks
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Replace final layers with noisy layers
        self.noisy_layer1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_layer2 = NoisyLinear(hidden_dim, action_dim)

    def forward(self, x):
        """Forward pass"""
        features = self.feature(x)
        x = F.relu(self.noisy_layer1(features))
        return self.noisy_layer2(x)

    def reset_noise(self):
        """Reset noise in all noisy layers"""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()

    def act(self, state):
        """Select action (greedy w.r.t. noisy Q-values)"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(1).item()
```

#### C. Training Algorithm

```python
class NoisyDQNAgent:
    """
    DQN Agent with Noisy Networks
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.q_network = NoisyDQN(state_dim, action_dim)
        self.target_network = NoisyDQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        """
        Select action without epsilon-greedy
        Exploration comes from weight noise
        """
        return self.q_network.act(state)

    def train_step(self, batch):
        """
        Training step

        Args:
            batch: (states, actions, rewards, next_states, dones)

        Returns:
            float: TD loss
        """
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Reset noise for both networks
        self.q_network.reset_noise()
        self.target_network.reset_noise()

        # Current Q-values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        # Loss
        loss = F.mse_loss(q_value, target_q_value)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

#### D. Advantages of Noisy Networks

1. **State-Dependent Exploration:** Noise affects Q-values differently in different states
2. **Learned Exploration Schedule:** Network learns when and how much to explore
3. **No Hyperparameter Tuning:** No need to tune $\epsilon$ or decay schedule
4. **Consistent Exploration:** Noise sampled once per episode maintains consistency

**Empirical Results:**

- Outperforms $\epsilon$-greedy on Atari games
- Particularly effective in environments requiring long-term planning
- Works well with prioritized experience replay

---

<a name="section-8"></a>

## 8. Bootstrap DQN

### Problem 8.1: Deep Exploration via Ensembles

**Question:** Implement Bootstrap DQN and explain how it achieves deep exploration.

**Solution:**

#### A. Theoretical Foundation

**Key Insight:** Maintain ensemble of Q-networks trained on different data subsets to capture epistemic uncertainty.

**Algorithm:**

1. Train $K$ independent Q-network heads $\{Q_1, ..., Q_K\}$
2. At episode start, randomly select active head $k \sim \text{Uniform}(\{1,...,K\})$
3. Follow policy $\pi_k(s) = \argmax_a Q_k(s,a)$ throughout episode
4. Update all heads on each transition (with bootstrap masks)

**Why Deep Exploration:**

- Different heads learn different policies
- Commitment to one policy per episode enables exploration of coherent strategies
- Uncertainty in ensemble drives exploration

#### B. Implementation

```python
class BootstrapHead(nn.Module):
    """
    Single Q-network head
    """
    def __init__(self, feature_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, features):
        return self.network(features)

class BootstrapDQN(nn.Module):
    """
    Bootstrap DQN with multiple heads
    """
    def __init__(self, state_dim, action_dim, num_heads=10,
                 feature_dim=128, hidden_dim=128):
        """
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension
            num_heads (int): Number of bootstrap heads
            feature_dim (int): Shared feature dimension
            hidden_dim (int): Hidden dimension for each head
        """
        super().__init__()

        self.num_heads = num_heads
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

        # Multiple heads
        self.heads = nn.ModuleList([
            BootstrapHead(feature_dim, action_dim, hidden_dim)
            for _ in range(num_heads)
        ])

        # Active head for current episode
        self.active_head = 0

    def forward(self, state, head_idx=None):
        """
        Forward pass

        Args:
            state: Input state
            head_idx: If None, return all heads; else return specific head

        Returns:
            Q-values from specified head(s)
        """
        features = self.feature_net(state)

        if head_idx is not None:
            return self.heads[head_idx](features)
        else:
            return torch.stack([head(features) for head in self.heads])

    def sample_head(self):
        """Sample active head for new episode"""
        self.active_head = np.random.randint(self.num_heads)
        return self.active_head

    def act(self, state):
        """Select action using active head"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor, self.active_head)
            return q_values.argmax(1).item()

class BootstrapDQNAgent:
    """
    Bootstrap DQN Agent
    """
    def __init__(self, state_dim, action_dim, num_heads=10,
                 lr=1e-3, gamma=0.99, mask_prob=0.5):
        """
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension
            num_heads (int): Number of bootstrap heads
            lr (float): Learning rate
            gamma (float): Discount factor
            mask_prob (float): Probability of updating each head
        """
        self.num_heads = num_heads
        self.mask_prob = mask_prob
        self.gamma = gamma

        self.q_network = BootstrapDQN(state_dim, action_dim, num_heads)
        self.target_network = BootstrapDQN(state_dim, action_dim, num_heads)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def begin_episode(self):
        """Sample new head for episode"""
        return self.q_network.sample_head()

    def select_action(self, state):
        """Select action using active head"""
        return self.q_network.act(state)

    def train_step(self, batch):
        """
        Training step with bootstrap sampling

        Args:
            batch: (states, actions, rewards, next_states, dones)

        Returns:
            list: Losses for each head
        """
        states, actions, rewards, next_states, dones = batch
        batch_size = len(states)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Generate bootstrap masks for each head
        masks = torch.bernoulli(
            torch.ones(self.num_heads, batch_size) * self.mask_prob
        )

        losses = []

        # Update each head
        for k in range(self.num_heads):
            # Current Q-values for head k
            q_values = self.q_network(states, head_idx=k)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q-values for head k
            with torch.no_grad():
                next_q_values = self.target_network(next_states, head_idx=k)
                next_q_value = next_q_values.max(1)[0]
                target_q_value = rewards + self.gamma * next_q_value * (1 - dones)

            # Masked loss (only update on bootstrapped samples)
            mask = masks[k]
            loss = (mask * (q_value - target_q_value) ** 2).sum() / (mask.sum() + 1e-8)
            losses.append(loss)

        # Total loss
        total_loss = sum(losses)

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return [loss.item() for loss in losses]

    def get_ensemble_uncertainty(self, state):
        """
        Compute uncertainty as variance across ensemble

        Args:
            state: Input state

        Returns:
            np.array: Uncertainty (std) for each action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            all_q_values = self.q_network(state_tensor)  # [num_heads, 1, action_dim]

            # Compute std across heads
            uncertainty = all_q_values.squeeze(1).std(dim=0).numpy()

        return uncertainty
```

#### C. Comparison with Other Methods

| Method            | Exploration Type | Within-Episode Consistency | Computational Cost |
| ----------------- | ---------------- | -------------------------- | ------------------ |
| **ε-greedy**      | Random           | No                         | Very Low           |
| **Noisy Nets**    | Learned          | Yes (per step)             | Low                |
| **Bootstrap DQN** | Uncertainty      | Yes (per episode)          | High (K networks)  |

#### D. Advantages of Bootstrap DQN

1. **Deep Exploration:** Committed exploration within episodes
2. **Epistemic Uncertainty:** Captures model uncertainty, not just noise
3. **Principled:** Based on Bayesian bootstrap
4. **No Hyperparameters:** No ε or noise parameters to tune

**Empirical Results (Osband et al., 2016):**

- Outperforms ε-greedy and Boltzmann on "Chain" environment
- More sample-efficient in problems requiring long-term planning
- Discovers diverse strategies through ensemble

---

<a name="section-9"></a>

## 9. Regret Analysis and Sample Complexity

### Problem 9.1: Theoretical Bounds

**Question:** Derive sample complexity bounds for different exploration strategies.

**Solution:**

#### A. PAC (Probably Approximately Correct) Framework

**Definition:** An algorithm is $(\epsilon, \delta)$-PAC if with probability at least $1-\delta$:
$$V^*(s_0) - V^\pi(s_0) \leq \epsilon$$

for all but polynomially many steps.

#### B. Sample Complexity of Exploration Strategies

**1. Random Exploration:**

**Theorem:** Random policy requires $\Omega(1/\epsilon^2)$ samples per state-action to estimate Q-values within $\epsilon$.

**Proof:** By Hoeffding's inequality, to ensure:
$$P(|\hat{Q}(s,a) - Q(s,a)| > \epsilon) < \delta$$

requires:
$$N(s,a) \geq \frac{2\log(2/\delta)}{\epsilon^2}$$

For $|S||A|$ state-action pairs:
$$\text{Total samples} = O\left(\frac{|S||A|}{\epsilon^2}\log\frac{|S||A|}{\delta}\right)$$

**2. UCB-Based Exploration:**

**Theorem (UCBVI):** For episodic MDPs with horizon $H$:
$$\text{Regret}(T) = \tilde{O}(\sqrt{H^3 |S| |A| T})$$

**Key Components:**

- Bonus: $\beta_t(s,a) = c\sqrt{\frac{H\log(|S||A|T/\delta)}{N_t(s,a)}}$
- Sample complexity: $\tilde{O}(\frac{H^3|S||A|}{\epsilon^2})$ to find $\epsilon$-optimal policy

**3. R-Max:**

**Theorem:** R-Max is $(\epsilon, \delta)$-PAC with sample complexity:
$$\tilde{O}\left(\frac{|S|^2|A|H^6}{\epsilon^3(1-\gamma)^6}\log\frac{1}{\delta}\right)$$

**Intuition:**

- Needs $m = O(\frac{H^2}{\epsilon^2(1-\gamma)^2})$ samples per $(s,a)$
- Optimistic initialization drives systematic exploration

**4. Posterior Sampling (Thompson Sampling for MDPs):**

**Theorem (PSRL):** For episodic MDPs:
$$\text{Regret}(T) = \tilde{O}(\sqrt{H^3 |S| |A| T})$$

matches UCB-based methods but often better empirically.

#### C. Lower Bounds

**Theorem (Jaksch et al., 2010):** For any algorithm in episodic MDPs:
$$\text{Regret}(T) = \Omega(\sqrt{H^2 |S| |A| T})$$

This shows UCBVI and PSRL are nearly minimax optimal!

**Proof Sketch:**

1. Construct hard MDP instance with $|S|$ states
2. Use information-theoretic argument: need $\Omega(\sqrt{T})$ samples to distinguish optimal policy
3. Each mistake costs $O(H)$ reward

#### D. Gap-Dependent Bounds

For problems with suboptimality gaps $\Delta_{sa} = V^*(s) - Q^*(s,a)$:

**UCB Bound:**
$$\text{Regret}(T) = O\left(\sum_{s,a: \Delta_{sa} > 0} \frac{H^2\log T}{\Delta_{sa}}\right)$$

Much better when gaps are large!

**Comparison:**

| Algorithm            | Regret (worst-case)    | Regret (gap-dependent) | Sample Complexity         |
| -------------------- | ---------------------- | ---------------------- | ------------------------- |
| **Random**           | $\Theta(T)$            | $\Theta(T)$            | $O(1/\epsilon^2)$         |
| **ε-greedy (decay)** | $O(\log T / \epsilon)$ | $O(\log T / \Delta)$   | $O(1/\epsilon^2)$         |
| **UCB**              | $\tilde{O}(\sqrt{T})$  | $O(\log T / \Delta)$   | $\tilde{O}(1/\epsilon^2)$ |
| **Thompson**         | $\tilde{O}(\sqrt{T})$  | $O(\log T / \Delta)$   | $\tilde{O}(1/\epsilon^2)$ |

---

<a name="section-10"></a>

## 10. Empirical Comparisons and Analysis

### Problem 10.1: Multi-Armed Bandit Testbed

**Question:** Implement a comprehensive comparison of bandit algorithms.

**Solution:**

#### A. Experimental Setup

```python
import matplotlib.pyplot as plt
import seaborn as sns

class BanditEnvironment:
    """
    Multi-Armed Bandit Environment
    """
    def __init__(self, num_arms=10, distribution='gaussian'):
        """
        Args:
            num_arms (int): Number of arms
            distribution (str): 'gaussian', 'bernoulli', or 'heavy-tail'
        """
        self.num_arms = num_arms
        self.distribution = distribution

        # Generate true means
        if distribution == 'gaussian':
            self.true_means = np.random.randn(num_arms)
        elif distribution == 'bernoulli':
            self.true_means = np.random.beta(2, 2, num_arms)
        elif distribution == 'heavy-tail':
            self.true_means = np.random.standard_t(3, num_arms)

        self.optimal_mean = np.max(self.true_means)
        self.optimal_arm = np.argmax(self.true_means)

    def pull(self, arm):
        """Pull arm and return reward"""
        if self.distribution == 'gaussian':
            return np.random.randn() + self.true_means[arm]
        elif self.distribution == 'bernoulli':
            return float(np.random.rand() < self.true_means[arm])
        elif distribution == 'heavy-tail':
            return np.random.standard_t(3) + self.true_means[arm]

    def regret(self, arm):
        """Instantaneous regret"""
        return self.optimal_mean - self.true_means[arm]

def run_bandit_experiment(algorithm_class, env, num_steps, num_runs=100, **kwargs):
    """
    Run bandit experiment multiple times

    Returns:
        dict: {
            'rewards': average rewards over time,
            'regrets': cumulative regret,
            'optimal_actions': fraction of optimal actions
        }
    """
    all_rewards = []
    all_regrets = []
    all_optimal = []

    for run in range(num_runs):
        # Initialize algorithm
        agent = algorithm_class(num_arms=env.num_arms, **kwargs)

        rewards = []
        regrets = []
        optimal_actions = []

        for t in range(num_steps):
            # Select action
            action = agent.select_action()

            # Get reward
            reward = env.pull(action)

            # Update agent
            agent.update(action, reward)

            # Record metrics
            rewards.append(reward)
            regrets.append(env.regret(action))
            optimal_actions.append(action == env.optimal_arm)

        all_rewards.append(rewards)
        all_regrets.append(np.cumsum(regrets))
        all_optimal.append(optimal_actions)

    return {
        'rewards': np.mean(all_rewards, axis=0),
        'regrets': np.mean(all_regrets, axis=0),
        'optimal_actions': np.mean(all_optimal, axis=0)
    }

def plot_bandit_comparison(results, num_steps):
    """
    Plot comparison of bandit algorithms

    Args:
        results (dict): {algorithm_name: experiment_results}
        num_steps (int): Number of steps
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot average reward
    ax = axes[0]
    for name, data in results.items():
        ax.plot(data['rewards'], label=name, alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot cumulative regret
    ax = axes[1]
    for name, data in results.items():
        ax.plot(data['regrets'], label=name, alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot optimal action percentage
    ax = axes[2]
    for name, data in results.items():
        # Smooth with moving average
        window = 100
        smoothed = np.convolve(
            data['optimal_actions'],
            np.ones(window)/window,
            mode='valid'
        )
        ax.plot(smoothed, label=name, alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Optimal Action Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

#### B. Run Experiments

```python
# Setup
num_arms = 10
num_steps = 10000
num_runs = 100

env = BanditEnvironment(num_arms=num_arms, distribution='gaussian')

# Algorithms to compare
algorithms = {
    'ε-greedy (0.1)': (EpsilonGreedy, {'epsilon': 0.1}),
    'ε-greedy (decay)': (DecayingEpsilonGreedy, {'c': 1.0, 'd': 0.1}),
    'UCB1': (UCB1, {'c': np.sqrt(2)}),
    'Thompson Sampling': (ThompsonSampling, {'alpha_prior': 1.0, 'beta_prior': 1.0}),
}

# Run experiments
results = {}
for name, (alg_class, kwargs) in algorithms.items():
    print(f"Running {name}...")
    results[name] = run_bandit_experiment(
        alg_class, env, num_steps, num_runs, **kwargs
    )

# Plot results
fig = plot_bandit_comparison(results, num_steps)
plt.savefig('bandit_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final statistics
print("\n=== Final Results (at T=10000) ===")
for name, data in results.items():
    final_regret = data['regrets'][-1]
    final_optimal_pct = data['optimal_actions'][-500:].mean() * 100
    print(f"{name:25s} | Regret: {final_regret:8.2f} | Optimal: {final_optimal_pct:5.1f}%")
```

#### C. Expected Results

**Typical Performance on 10-arm Gaussian Bandit:**

| Algorithm         | Final Regret | % Optimal (last 500) | Notes                     |
| ----------------- | ------------ | -------------------- | ------------------------- |
| ε-greedy (0.1)    | 800-1000     | 85-90%               | Linear regret, but simple |
| ε-greedy (decay)  | 50-100       | 95-98%               | Better with tuning        |
| UCB1              | 30-50        | 98-99%               | Near-optimal, no tuning   |
| Thompson Sampling | 25-40        | 98-99%               | Best overall, adaptive    |

**Key Observations:**

1. UCB and Thompson Sampling achieve logarithmic regret
2. Fixed ε-greedy has linear regret (keeps exploring)
3. Thompson Sampling adapts better to problem structure
4. All algorithms eventually identify optimal arm

### Problem 10.2: Deep RL Exploration Comparison

**Question:** Compare exploration methods in deep RL on hard exploration environments.

**Solution:**

#### A. Environment: Sparse Reward Grid World

```python
class SparseGridWorld:
    """
    Grid world with sparse reward requiring exploration
    """
    def __init__(self, size=10, goal_pos=None):
        self.size = size
        self.goal_pos = goal_pos or (size-1, size-1)
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        return self._get_state()

    def _get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.agent_pos] = 1
        return state.flatten()

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1,0), (0,1), (1,0), (0,-1)]
        dx, dy = moves[action]

        new_x = np.clip(self.agent_pos[0] + dx, 0, self.size-1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, self.size-1)
        self.agent_pos = (new_x, new_y)

        # Sparse reward only at goal
        reward = 1.0 if self.agent_pos == self.goal_pos else 0.0
        done = self.agent_pos == self.goal_pos

        return self._get_state(), reward, done

# Experimental comparison
def compare_deep_exploration(env, algorithms, num_episodes=1000):
    """
    Compare deep RL exploration methods

    Args:
        env: Environment
        algorithms: Dict of {name: agent_class}
        num_episodes: Number of training episodes

    Returns:
        dict: Results for each algorithm
    """
    results = {}

    for name, agent_class in algorithms.items():
        print(f"\nTraining {name}...")
        agent = agent_class()

        episode_rewards = []
        episode_lengths = []
        success_rate = []

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0

            while steps < 200:  # Max episode length
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)

                agent.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            # Success rate (moving average)
            success = total_reward > 0
            if episode < 100:
                success_rate.append(success)
            else:
                success_rate.append(
                    0.99 * success_rate[-1] + 0.01 * success
                )

        results[name] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'success_rate': success_rate
        }

    return results
```

#### B. Expected Results on Hard Exploration

**Success Rate After 1000 Episodes:**

| Method             | Success Rate | Episodes to First Success | Final Performance |
| ------------------ | ------------ | ------------------------- | ----------------- |
| **Random**         | 5-10%        | ~500                      | Poor              |
| **ε-greedy (0.1)** | 40-50%       | ~200                      | Moderate          |
| **Count-Based**    | 80-90%       | ~50                       | Good              |
| **RND**            | 90-95%       | ~30                       | Very Good         |
| **Noisy Nets**     | 85-90%       | ~40                       | Very Good         |
| **Bootstrap DQN**  | 90-95%       | ~35                       | Very Good         |

**Key Insights:**

1. **Count-based and intrinsic motivation** crucial for sparse rewards
2. **RND** particularly effective in high-dimensional spaces
3. **Bootstrap DQN** provides systematic exploration
4. **ε-greedy** alone often insufficient for hard exploration

---

## Conclusion

This comprehensive document has presented detailed solutions to HW8 on Exploration Methods in Reinforcement Learning, covering both theoretical foundations and practical implementations.

### Summary of Key Findings

**1. Multi-Armed Bandits:**

- Established fundamental exploration-exploitation trade-off
- UCB1 and Thompson Sampling achieve $O(\log T)$ regret
- Thompson Sampling often superior empirically due to natural Bayesian exploration

**2. Exploration in MDPs:**

- Count-based methods provide principled exploration through optimism
- R-Max achieves PAC guarantees with sample complexity $\tilde{O}(|S|^2|A|H^6/\epsilon^3)$
- Exploration bonuses ($\beta/\sqrt{N}$) theoretically motivated by concentration inequalities

**3. Deep RL Exploration:**

- **Intrinsic Motivation** (RND, ICM) enables exploration in high-dimensional spaces
- **Noisy Networks** provide state-dependent, learnable exploration
- **Bootstrap DQN** captures epistemic uncertainty through ensembles

**4. Theoretical Guarantees:**

- Lower bound: Any algorithm has $\Omega(\sqrt{KT})$ regret on some bandit instance
- UCB-based methods nearly minimax optimal with $\tilde{O}(\sqrt{T})$ regret
- Gap-dependent bounds: $O(\log T / \Delta)$ achievable when suboptimality gaps are large

**5. Empirical Insights:**

- Simple $\epsilon$-greedy sufficient for easy exploration problems
- Hard exploration (sparse rewards) requires intrinsic motivation or count-based methods
- RND particularly effective in pixel-based environments (e.g., Montezuma's Revenge)
- Thompson Sampling consistently strong across diverse problem settings

### Practical Recommendations

**For Tabular MDPs:**

1. Use UCB-based exploration with confidence bounds
2. Consider R-Max for PAC guarantees
3. Thompson Sampling if Bayesian prior available

**For Deep RL:**

1. Start with $\epsilon$-greedy or noisy networks as baseline
2. Add RND for hard exploration problems
3. Consider count-based methods for discrete state spaces
4. Use Bootstrap DQN when epistemic uncertainty critical

**Hyperparameter Guidelines:**

- $\epsilon$-greedy: $\epsilon \in [0.01, 0.1]$, decay over time
- UCB: $c = \sqrt{2}$ (theoretical optimum)
- RND: Scale intrinsic rewards to match extrinsic magnitude
- Count bonus: $\beta \approx 0.1$ to $1.0$ depending on reward scale

### Open Research Questions

1. **Sample efficiency:** Can we achieve better than $\sqrt{T}$ regret for general MDPs?
2. **Scalability:** How to efficiently count in continuous high-dimensional spaces?
3. **Noisy TV problem:** Better solutions beyond feature learning?
4. **Transfer:** Can exploration strategies transfer across tasks?
5. **Multi-task:** Optimal exploration when learning multiple related MDPs?

### Future Directions

**Emerging Methods:**

- **Never Give Up (NGU):** Combines episodic and lifelong novelty
- **Go-Explore:** Returns to promising states, explores systematically
- **Hindsight Experience Replay:** Creates useful experiences from failures
- **Successor Features:** Generalize value functions for exploration
- **Maximum Entropy RL:** Implicitly explores through entropy maximization

**Theoretical Advances:**

- Tighter instance-dependent bounds
- Better understanding of deep exploration
- Sample complexity in function approximation setting
- Exploration in partially observable environments

### Implementation Checklist

For implementing exploration in your RL agent:

- [ ] Choose exploration strategy based on problem difficulty
- [ ] Implement baseline ($\epsilon$-greedy or Boltzmann)
- [ ] Add intrinsic motivation if sparse rewards present
- [ ] Monitor visitation counts/frequencies
- [ ] Track exploration vs exploitation balance
- [ ] Log regret or sample efficiency metrics
- [ ] Visualize state space coverage
- [ ] Tune exploration hyperparameters systematically
- [ ] Compare multiple exploration strategies
- [ ] Implement early stopping when exploration plateaus

### Acknowledgments

This work builds upon decades of research in exploration-exploitation dilemmas, from the foundational work of Robbins (1952) on multi-armed bandits to recent advances in deep reinforcement learning exploration. The implementations provided draw from open-source libraries and research papers, adapted for educational purposes.

---

## References

### Core Papers

[1] **Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).** "Finite-time Analysis of the Multiarmed Bandit Problem." _Machine Learning_, 47(2-3), 235-256. DOI: 10.1023/A:1013689704352

[2] **Agrawal, S., & Goyal, N. (2012).** "Analysis of Thompson Sampling for the Multi-armed Bandit Problem." _Conference on Learning Theory (COLT)_, 39.1-39.26.

[3] **Lai, T. L., & Robbins, H. (1985).** "Asymptotically Efficient Adaptive Allocation Rules." _Advances in Applied Mathematics_, 6(1), 4-22.

[4] **Thompson, W. R. (1933).** "On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples." _Biometrika_, 25(3/4), 285-294.

### Count-Based and Model-Based Methods

[5] **Brafman, R. I., & Tennenholtz, M. (2002).** "R-MAX - A General Polynomial Time Algorithm for Near-Optimal Reinforcement Learning." _Journal of Machine Learning Research_, 3, 213-231.

[6] **Kolter, J. Z., & Ng, A. Y. (2009).** "Near-Bayesian Exploration in Polynomial Time." _International Conference on Machine Learning (ICML)_, 513-520.

[7] **Strehl, A. L., & Littman, M. L. (2008).** "An Analysis of Model-Based Interval Estimation for Markov Decision Processes." _Journal of Computer and System Sciences_, 74(8), 1309-1331.

[8] **Bellemare, M., Srinivasan, S., Ostrovski, G., et al. (2016).** "Unifying Count-Based Exploration and Intrinsic Motivation." _Advances in Neural Information Processing Systems (NIPS)_, 1471-1479.

### Intrinsic Motivation

[9] **Schmidhuber, J. (1991).** "Curious Model-Building Control Systems." _IEEE International Joint Conference on Neural Networks_, 1458-1463.

[10] **Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).** "Curiosity-driven Exploration by Self-supervised Prediction." _International Conference on Machine Learning (ICML)_, 2778-2787.

[11] **Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).** "Exploration by Random Network Distillation." _International Conference on Learning Representations (ICLR)_.

[12] **Stadie, B. C., Levine, S., & Abbeel, P. (2015).** "Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models." _arXiv preprint arXiv:1507.00814_.

### Deep RL Exploration

[13] **Fortunato, M., Azar, M. G., Piot, B., et al. (2018).** "Noisy Networks for Exploration." _International Conference on Learning Representations (ICLR)_.

[14] **Osband, I., Blundell, C., Pritzel, A., & Van Roy, B. (2016).** "Deep Exploration via Bootstrapped DQN." _Advances in Neural Information Processing Systems (NIPS)_, 4026-4034.

[15] **Plappert, M., Houthooft, R., Dhariwal, P., et al. (2018).** "Parameter Space Noise for Exploration." _International Conference on Learning Representations (ICLR)_.

[16] **Ecoffet, A., Huizinga, J., Lehman, J., et al. (2021).** "First Return, Then Explore." _Nature_, 590, 580-586.

### Theoretical Foundations

[17] **Jaksch, T., Ortner, R., & Auer, P. (2010).** "Near-optimal Regret Bounds for Reinforcement Learning." _Journal of Machine Learning Research_, 11, 1563-1600.

[18] **Azar, M. G., Osband, I., & Munos, R. (2017).** "Minimax Regret Bounds for Reinforcement Learning." _International Conference on Machine Learning (ICML)_, 263-272.

[19] **Jin, C., Allen-Zhu, Z., Bubeck, S., & Jordan, M. I. (2018).** "Is Q-Learning Provably Efficient?" _Advances in Neural Information Processing Systems (NeurIPS)_, 4863-4873.

### Books and Surveys

[20] **Sutton, R. S., & Barto, A. G. (2018).** _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. [Online: http://incompleteideas.net/book/the-book.html]

[21] **Lattimore, T., & Szepesvári, C. (2020).** _Bandit Algorithms_. Cambridge University Press. [Online: https://tor-lattimore.com/downloads/book/book.pdf]

[22] **Szepesvári, C. (2010).** _Algorithms for Reinforcement Learning_. Morgan & Claypool Publishers.

[23] **Arora, S., & Doshi, P. (2021).** "A Survey of Inverse Reinforcement Learning: Techniques, Methods, and Applications." _Expert Systems with Applications_, 186, 115672.

### Additional Resources

[24] **OpenAI Spinning Up.** [https://spinningup.openai.com/](https://spinningup.openai.com/)

[25] **Deep RL Course (Hugging Face).** [https://huggingface.co/deep-rl-course](https://huggingface.co/deep-rl-course)

[26] **DeepMind x UCL RL Lecture Series.** [https://deepmind.com/learning-resources](https://deepmind.com/learning-resources)

---

## Appendix: Code Repository

All code implementations from this document are available at:

- [GitHub Repository] (placeholder)
- Includes: Multi-armed bandits, exploration algorithms, deep RL implementations
- Dependencies: `numpy`, `torch`, `gym`, `matplotlib`, `seaborn`
- Installation: `pip install -r requirements.txt`

### Running Experiments

```bash
# Multi-armed bandit comparison
python bandits/compare_algorithms.py --num_arms 10 --steps 10000

# Deep RL exploration
python deep_rl/train_exploration.py --env GridWorld --method RND

# Reproduce all figures
python visualize/generate_plots.py --output figures/
```

---

## Appendix: Mathematical Notation

| Symbol        | Meaning                                    |
| ------------- | ------------------------------------------ | --- | ----------------- |
| $K$           | Number of arms/actions                     |
| $\mu_i$       | True mean reward of arm $i$                |
| $\hat{\mu}_i$ | Empirical mean reward of arm $i$           |
| $N_i(t)$      | Number of times arm $i$ pulled by time $t$ |
| $R(T)$        | Cumulative regret after $T$ steps          |
| $\Delta_i$    | Suboptimality gap: $\mu^* - \mu_i$         |
| $\delta$      | Confidence parameter (typically 0.05)      |
| $\epsilon$    | Approximation error                        |
| $\gamma$      | Discount factor                            |
| $H$           | Horizon (episode length)                   |
| $             | S                                          | $   | Number of states  |
| $             | A                                          | $   | Number of actions |
| $Q(s,a)$      | Action-value function                      |
| $V(s)$        | State-value function                       |
| $\pi$         | Policy                                     |
| $\theta$      | Parameter vector                           |

---

**Document Information:**

- **Total Pages:** ~45
- **Word Count:** ~12,000
- **Code Blocks:** 30+
- **Equations:** 100+
- **Figures:** 10+ (referenced)
- **Tables:** 15+

**Compiled:** 2024  
**Format:** IEEE Standard  
**License:** Educational Use

---

**End of Document**
