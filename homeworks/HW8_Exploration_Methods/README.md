# HW8: Exploration Methods in RL

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Exploration](https://img.shields.io/badge/Topic-Exploration-purple.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment explores the exploration-exploitation dilemma in reinforcement learning. Effective exploration is crucial for discovering high-reward states and actions, especially in environments with sparse rewards or deceptive local optima.

## 🎯 Learning Objectives

1. **Exploration vs Exploitation**: Understand the fundamental trade-off
2. **Multi-Armed Bandits**: Master exploration strategies in stateless setting
3. **Intrinsic Motivation**: Learn count-based and prediction-based exploration
4. **Upper Confidence Bounds**: Understand optimism under uncertainty
5. **Thompson Sampling**: Master Bayesian exploration strategies
6. **Exploration Bonuses**: Design and implement exploration incentives
7. **Deep Exploration**: Learn methods for exploration in deep RL

## 📂 Directory Structure

```
HW8_Exploration_Methods/
├── code/                       # (For this assignment, implement if required)
│   ├── HW8_Notebook.ipynb      # Structured workflow: config → data → policies → training → viz → metadata
│   └── requirements.txt        # Minimal deps for the notebook
├── answers/                    # (Submit written/coded answers)
├── reports/
│   └── HW8_Report.tex          # IEEE-style write-up aligned to the notebook
│   └── HW8_Questions.pdf      # Assignment questions
└── README.md
```

### 🔧 How to run the provided notebook

1. Create a virtual environment in this folder and install dependencies:

```bash
cd homeworks/HW8_Exploration_Methods/code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Open `HW8_Notebook.ipynb` and run cells top-down. The first code cell centralizes seeds, device, and hyperparameters. A `run_info.json` file is written to capture commit hash, hardware, and config.
3. Regret plots are saved under `homeworks/pictures/hw8_regret.png` (directory is auto-created).

## Command-line helpers

Two utility scripts are provided for running and visualizing the canonical bandit experiments from the notebook:

- `code/run_bandits.py` — run Bernoulli bandit experiments (ε-greedy, UCB1, Thompson), average results over multiple runs, and save cumulative-regret plot.
- `bandits/compare_algorithms.py` — wrapper that calls the runner, saves compressed results (`.npz`) under `homeworks/HW8_Exploration_Methods/results/`, and invokes the plotting helper.
- `visualize/generate_plots.py` — loads the `.npz` produced by the previous script and writes `bandit_avg_rewards.png` and `bandit_cum_regret.png` to an output folder (default `homeworks/HW8_Exploration_Methods/pictures/`).

Quick examples:

```bash
# from repo root
cd homeworks/HW8_Exploration_Methods
python code/run_bandits.py --num_arms 10 --steps 5000 --runs 20 --out pictures/bandit_comparison.png

# or use the comparison wrapper (saves results.npz and runs visualizer)
python bandits/compare_algorithms.py --num_arms 10 --steps 5000 --runs 50 --out_dir results

# generate plots directly from saved .npz
python visualize/generate_plots.py --results results/bandit_results.npz --out_dir pictures
```

These commands are optional; the notebook provides the same functionality but the scripts are useful for batch runs or CI.

### 📝 Report alignment

- `reports/HW8_Report.tex` uses the IEEEtran class and summarizes the notebook’s Bernoulli bandit baselines (ε-greedy, UCB1, Thompson). Compile with `pdflatex` (twice) after running the notebook to refresh figures and tables.

## 📚 Core Concepts

### 1. The Exploration-Exploitation Dilemma

**Fundamental Trade-off:**

```
Exploration: Try new actions to discover better strategies
Exploitation: Use known information to maximize immediate reward
```

**Examples:**

- **Restaurant**: Try new restaurant (explore) vs go to favorite (exploit)
- **A/B Testing**: Test new webpage (explore) vs use best known (exploit)
- **Clinical Trials**: Try experimental treatment vs use standard care

**Formal Problem:**

```
πt = argmax 𝔼[∑τ=t^T r(sτ, aτ) | πt, history]
      π
```

Challenge: Future rewards depend on information gathered through exploration.

### 2. Multi-Armed Bandits (MAB)

**Simplest Exploration Setting:** Stateless RL

**Problem Formulation:**

```
K arms (actions)
Each arm i has reward distribution with mean μi
Goal: Maximize cumulative reward = minimize regret

Regret: R(T) = T·μ* - ∑t=1^T rt

where μ* = max μi (best arm's mean)
           i
```

**Key Concepts:**

- **Regret**: Lost reward compared to always picking best arm
- **Sample Complexity**: How many pulls to identify best arm
- **Sublinear Regret**: R(T) = o(T) means eventually exploiting well

### 3. Exploration Strategies

#### a) ε-Greedy

```python
def epsilon_greedy(Q, epsilon):
    if random() < epsilon:
        return random_action()  # Explore
    else:
        return argmax(Q)  # Exploit
```

**Properties:**

- Simple to implement
- Wastes exploration on suboptimal arms
- Linear regret: R(T) = O(T)
- Decaying ε → sublinear regret

#### b) Upper Confidence Bound (UCB)

```python
def ucb1(counts, values, t, c=√2):
    """
    UCB1 algorithm
    counts[i]: times arm i pulled
    values[i]: average reward from arm i
    t: total timesteps
    c: exploration constant
    """
    ucb_values = values + c * sqrt(log(t) / (counts + 1e-8))
    return argmax(ucb_values)
```

**Intuition:** "Optimism in the face of uncertainty"

- Confidence bound decreases with more samples
- Encourages trying uncertain arms
- Regret bound: R(T) = O(√(T log T))

**Mathematical Justification:**

```
With probability ≥ 1-δ:
|Q̂(a) - Q(a)| ≤ √(2 log(1/δ) / N(a))

where N(a) = count of action a
```

#### c) Thompson Sampling (Bayesian)

```python
def thompson_sampling(alpha, beta):
    """
    Bayesian approach using Beta distribution
    alpha[i], beta[i]: parameters for arm i
    """
    # Sample from posterior distribution
    samples = [random.beta(alpha[i], beta[i])
               for i in range(K)]
    return argmax(samples)

# Update after observing reward r ∈ {0,1}
def update(arm, reward, alpha, beta):
    if reward == 1:
        alpha[arm] += 1
    else:
        beta[arm] += 1
```

**Intuition:** Sample from belief distribution

- Naturally balances exploration/exploitation
- Optimal Bayesian strategy
- Empirically strong performance

**Properties:**

- Regret: R(T) = O(√(KT log T))
- Adapts to problem difficulty
- Works well empirically

### 4. Count-Based Exploration

**Key Idea:** Bonus for visiting rarely-seen states

#### Exploration Bonus

```python
def exploration_bonus(state, counts, beta=0.1):
    """
    Add bonus proportional to 1/√count
    """
    N = counts[state]
    bonus = beta / sqrt(N + 1)
    return bonus
```

**R-Max Algorithm:**

```
Initialize all Q(s,a) to Rmax (optimistic)
Update Q-values as usual, but only after
state-action visited sufficiently often

Result: Explore until confident, then exploit
```

**Advantages:**

- Theoretically motivated (PAC bounds)
- Simple to implement in tabular case
- Guarantees near-optimal policy

**Challenges in Deep RL:**

- How to count in continuous/high-dimensional spaces?
- Hash-based methods (SimHash, LSH)
- Learned density models

### 5. Intrinsic Motivation

**Philosophy:** Rewards for learning, not just external goals

#### a) Prediction Error (Forward Model)

```python
class ForwardModelBonus:
    def __init__(self, model):
        self.model = model  # Learns s,a → s'

    def intrinsic_reward(self, s, a, s_next):
        s_pred = self.model(s, a)
        error = ||s_pred - s_next||²
        return η * error  # η is scaling factor
```

**Intuition:** Novel states are hard to predict

#### b) Random Network Distillation (RND)

```python
class RND:
    def __init__(self):
        self.target = random_network()  # Fixed
        self.predictor = trainable_network()

    def intrinsic_reward(self, state):
        target_feat = self.target(state)
        pred_feat = self.predictor(state)
        return ||pred_feat - target_feat||²
```

**Why It Works:**

- Familiar states → predictor learns → low error → low bonus
- Novel states → high error → high bonus → exploration

#### c) Curiosity-Driven (ICM)

Combines forward and inverse models (see HW6 for details)

### 6. Deep Exploration Strategies

#### a) Noisy Networks

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x):
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        bias = self.bias_mu + self.bias_sigma * bias_epsilon

        return F.linear(x, weight, bias)
```

**Benefits:**

- State-dependent exploration
- Learned exploration schedule
- Parameters control exploration

#### b) Parameter Space Noise

```python
def add_parameter_noise(policy, std):
    """
    Add noise to network parameters
    """
    with torch.no_grad():
        for param in policy.parameters():
            param.add_(torch.randn_like(param) * std)
```

**Advantage:** Consistent exploration within episode

#### c) Bootstrap DQN

```python
class BootstrapDQN:
    def __init__(self, num_heads=10):
        self.heads = [QNetwork() for _ in range(num_heads)]

    def forward(self, state):
        # Each head gives Q-value estimate
        q_values = [head(state) for head in self.heads]
        return q_values

    def select_action(self, state):
        # Active head for this episode
        head_idx = random.randint(0, num_heads-1)
        q_vals = self.heads[head_idx](state)
        return argmax(q_vals)
```

**Intuition:** Uncertainty through ensemble → deep exploration

### 7. Comparison of Methods

| Method                | Sample Efficiency | Computational Cost | Theory           | Works in Deep RL |
| --------------------- | ----------------- | ------------------ | ---------------- | ---------------- |
| **ε-Greedy**          | Low               | Very Low           | Weak             | Yes              |
| **UCB**               | Medium            | Low                | Strong (bandits) | Difficult        |
| **Thompson Sampling** | High              | Medium             | Strong           | Difficult        |
| **Count-Based**       | High              | Medium             | Strong           | Needs tricks     |
| **RND**               | Medium            | Medium             | Weak             | Yes              |
| **Noisy Nets**        | Medium            | Low                | Weak             | Yes              |

## 📊 Topics Covered

1. **Multi-Armed Bandits**

   - Regret analysis
   - UCB algorithms
   - Thompson sampling
   - Contextual bandits

2. **Exploration Bonuses**

   - Count-based methods
   - MBIE-EB, R-Max
   - Pseudo-counts

3. **Intrinsic Motivation**

   - Prediction error
   - Information gain
   - Empowerment

4. **Deep Exploration**

   - Noisy networks
   - Bootstrap DQN
   - Parameter noise

5. **Theoretical Analysis**
   - Regret bounds
   - Sample complexity
   - PAC guarantees

## 📖 Key References

### Papers

1. **Auer, P., et al. (2002)** - "Finite-time Analysis of the Multiarmed Bandit Problem" - Machine Learning

2. **Bellemare, M., et al. (2016)** - "Unifying Count-Based Exploration and Intrinsic Motivation" - NIPS

3. **Burda, Y., et al. (2018)** - "Exploration by Random Network Distillation" - ICLR

4. **Fortunato, M., et al. (2018)** - "Noisy Networks for Exploration" - ICLR

5. **Osband, I., et al. (2016)** - "Deep Exploration via Bootstrapped DQN" - NIPS

6. **Pathak, D., et al. (2017)** - "Curiosity-driven Exploration by Self-supervised Prediction" - ICML

### Books

1. **Lattimore, T., & Szepesvári, C. (2020)** - _Bandit Algorithms_ - [Free Online](https://tor-lattimore.com/downloads/book/book.pdf)

2. **Sutton & Barto (2018)** - Chapter 2 (Multi-armed Bandits)

## 💡 Discussion Questions

1. **Why does ε-greedy have linear regret while UCB has logarithmic regret?**

2. **How does Thompson sampling balance exploration and exploitation without explicit exploration parameter?**

3. **What are the challenges of applying count-based exploration in high-dimensional spaces?**

4. **When might prediction-error-based intrinsic motivation fail? (Noisy-TV problem)**

5. **How do noisy networks differ from action-space noise (e.g., ε-greedy)?**

6. **Why is deep exploration (within-episode consistency) important?**

## 🎓 Extensions

- Implement multi-armed bandit testbed
- Compare exploration strategies on hard exploration games
- Analyze regret bounds empirically
- Try Never Give Up (NGU) algorithm
- Implement Go-Explore for hard exploration

---

**Course:** Deep Reinforcement Learning  
**Assignment Type:** Theory + Implementation  
**Last Updated:** 2024
