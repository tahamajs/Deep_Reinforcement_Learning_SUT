# HW2: Value-Based Methods in Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Value-Based](https://img.shields.io/badge/Methods-Value--Based-green.svg)](https://www.deepmind.com/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment explores **value-based reinforcement learning methods**, which learn the value of states or state-action pairs to derive optimal policies. These methods form the foundation of modern deep reinforcement learning and include both classical tabular methods and deep learning approaches.

## 🎯 Learning Objectives

By completing this assignment, you will:

1. **Understand Temporal Difference Learning**: Learn how TD methods combine ideas from Monte Carlo and Dynamic Programming
2. **Master On-Policy vs Off-Policy**: Distinguish between learning from current policy (SARSA) vs learning from optimal policy (Q-Learning)
3. **Deep Q-Networks (DQN)**: Implement neural network-based value functions for complex state spaces
4. **Address Overestimation Bias**: Understand and implement Double DQN to mitigate Q-value overestimation
5. **Analyze Convergence**: Study convergence properties and stability of different algorithms

## 📂 Directory Structure

```
HW2_Value_Based_Methods/
├── code/
│   ├── HW2_P1_SARSA_and_QLearning.ipynb    # Tabular TD methods
│   └── HW2_P2_DQN_vs_DDQN.ipynb             # Deep Q-Networks
├── answers/
│   ├── HW2_P1_SARSA_and_QLearning_Solution.ipynb
│   ├── HW2_P2_DQN_vs_DDQN_Solution.ipynb
│   └── HW2_Solution.pdf                      # Complete written solutions
├── reports/
│   └── HW2_Questions.pdf                     # Assignment questions
└── README.md
```

## 📚 Theoretical Background

### 1. Temporal Difference Learning

**Temporal Difference (TD) Learning** is a fundamental RL technique that learns value functions by bootstrapping from current estimates. Unlike Monte Carlo methods that wait until the end of an episode, TD methods update estimates online.

**Key Equation:**

```
V(St) ← V(St) + α[Rt+1 + γV(St+1) - V(St)]
                    └─────┬─────┘
                     TD Error (δt)
```

**Advantages:**

- Can learn from incomplete sequences
- Lower variance than Monte Carlo
- Suitable for continuing tasks

### 2. SARSA (On-Policy TD Control)

**SARSA** stands for State-Action-Reward-State-Action, representing the tuple (St, At, Rt+1, St+1, At+1) used in updates.

**Update Rule:**

```
Q(St, At) ← Q(St, At) + α[Rt+1 + γQ(St+1, At+1) - Q(St, At)]
```

**Key Characteristics:**

- **On-Policy**: Learns Q-values for the policy being followed
- **Conservative**: Accounts for exploration during learning
- **Safe**: Better for risky environments (e.g., cliff walking)

**Algorithm:**

1. Initialize Q(s,a) arbitrarily
2. For each episode:
   - Choose action At from St using policy derived from Q (e.g., ε-greedy)
   - Take action At, observe Rt+1, St+1
   - Choose action At+1 from St+1 using same policy
   - Update: Q(St,At) ← Q(St,At) + α[Rt+1 + γQ(St+1,At+1) - Q(St,At)]

### 3. Q-Learning (Off-Policy TD Control)

**Q-Learning** learns the optimal action-value function Q\* independent of the policy being followed.

**Update Rule:**

```
Q(St, At) ← Q(St, At) + α[Rt+1 + γ max Q(St+1, a) - Q(St, At)]
                                    a
```

**Key Characteristics:**

- **Off-Policy**: Can learn optimal policy while following exploratory policy
- **Aggressive**: Assumes optimal actions will be taken
- **Flexible**: Separates behavior policy from target policy

**Convergence Guarantee:** Under certain conditions (all state-action pairs visited infinitely often, learning rate decay), Q-Learning converges to Q\* with probability 1.

### 4. Deep Q-Networks (DQN)

**DQN** extends Q-Learning to high-dimensional state spaces using deep neural networks as function approximators.

**Key Innovations:**

#### a) Experience Replay

Stores transitions (s, a, r, s') in replay buffer D and samples mini-batches randomly.

**Benefits:**

- Breaks correlation between consecutive samples
- Improves sample efficiency through reuse
- Enables mini-batch SGD

#### b) Target Network

Uses a separate network Q̂ for generating targets, updated periodically.

**Loss Function:**

```
L(θ) = 𝔼[(r + γ max Q̂(s', a'; θ⁻) - Q(s, a; θ))²]
              a'
```

where θ⁻ are the target network parameters, updated every C steps: θ⁻ ← θ

**Why It Works:**

- Reduces correlation between target and predicted Q-values
- Stabilizes training by providing consistent targets
- Prevents harmful positive feedback loops

### 5. Double DQN (DDQN)

**Problem with DQN:** The max operator in standard Q-Learning leads to overestimation bias because it uses the same values for both selecting and evaluating actions.

**Solution:**

```
Q(St, At) ← Q(St, At) + α[Rt+1 + γQ(St+1, argmax Q(St+1, a; θ); θ⁻) - Q(St, At)]
                                         a
```

**Key Idea:**

- Use **online network (θ)** to select the best action
- Use **target network (θ⁻)** to evaluate that action's value

**Benefits:**

- Reduces overestimation bias
- More accurate value estimates
- Improved stability and performance
- Minimal computational overhead

## 💻 Implementation Details

### Part 1: SARSA vs Q-Learning

**Environment:** Typically GridWorld or similar tabular environment

**Tasks:**

1. Implement tabular SARSA algorithm
2. Implement tabular Q-Learning algorithm
3. Compare convergence rates and final policies
4. Analyze behavior in deterministic vs stochastic environments
5. Test on "dangerous" environments (e.g., CliffWalking)

**Expected Observations:**

- SARSA learns safer policies in risky environments
- Q-Learning finds optimal policies but may be riskier during learning
- Both converge given sufficient exploration

### Part 2: DQN vs DDQN

**Environment:** Atari games or continuous state spaces (CartPole, LunarLander)

**Neural Network Architecture:**

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**Hyperparameters:**

- Learning rate: α = 0.001
- Discount factor: γ = 0.99
- Batch size: 32-64
- Replay buffer size: 10,000-100,000
- Target network update frequency: Every 100-1000 steps
- ε-greedy: ε = 1.0 → 0.01 (linear decay)

**Tasks:**

1. Implement experience replay buffer
2. Implement DQN with target network
3. Implement DDQN modification
4. Compare training curves and final performance
5. Analyze Q-value estimates (overestimation in DQN vs DDQN)

## 📊 Evaluation Metrics

1. **Average Return**: Mean cumulative reward per episode
2. **Learning Curve**: Performance over training episodes
3. **Q-Value Statistics**: Mean/max Q-values to detect overestimation
4. **Convergence Speed**: Episodes to reach threshold performance
5. **Policy Quality**: Success rate in test episodes
6. **Stability**: Variance in performance across runs

## 🔧 Requirements

```python
# Core Libraries
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# RL Environments
gymnasium>=0.28.0  # (formerly gym)
ale-py>=0.8.0      # For Atari environments

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Utilities
pandas>=1.3.0
tqdm>=4.62.0
tensorboard>=2.9.0  # For logging
```

## 🚀 Getting Started

### Installation

```bash
# Clone repository and navigate to homework directory
cd HW2_Value_Based_Methods

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Code

1. **Part 1 - SARSA & Q-Learning:**

   ```bash
   jupyter notebook code/HW2_P1_SARSA_and_QLearning.ipynb
   ```

2. **Part 2 - DQN & DDQN:**
   ```bash
   jupyter notebook code/HW2_P2_DQN_vs_DDQN.ipynb
   ```

### Expected Runtime

- Part 1 (Tabular): ~5-10 minutes
- Part 2 (DQN/DDQN): ~30-60 minutes (depending on environment and episodes)

## 📈 Expected Results

### SARSA vs Q-Learning

- **Q-Learning**: Finds optimal path, higher variance during training
- **SARSA**: Learns safer path, more stable convergence
- **CliffWalking**: SARSA avoids cliff edge, Q-Learning takes risky optimal path

### DQN vs DDQN

- **DQN**: May show Q-value overestimation, less stable
- **DDQN**: More accurate Q-values, improved stability and performance
- **Performance**: DDQN typically achieves 10-30% better final performance

## 🐛 Common Issues & Solutions

### Issue 1: Slow Convergence

**Solution:**

- Increase learning rate (but ensure stability)
- Tune exploration schedule (ε-decay)
- Check replay buffer size

### Issue 2: Catastrophic Forgetting

**Solution:**

- Increase replay buffer size
- Reduce learning rate
- Update target network less frequently

### Issue 3: Overestimation Not Visible

**Solution:**

- Log Q-values during training
- Compare Q-values with actual returns
- Use environments with longer horizons

## 📖 References

### Seminal Papers

1. **Watkins, C. J., & Dayan, P. (1992)**

   - "Q-learning"
   - _Machine Learning, 8_(3-4), 279-292

2. **Rummery, G. A., & Niranjan, M. (1994)**

   - "On-line Q-learning using connectionist systems"
   - _Technical Report, Cambridge University_

3. **Mnih, V., et al. (2015)**

   - "Human-level control through deep reinforcement learning"
   - _Nature, 518_(7540), 529-533
   - [Paper](https://www.nature.com/articles/nature14236)

4. **Van Hasselt, H., Guez, A., & Silver, D. (2016)**
   - "Deep reinforcement learning with double q-learning"
   - _AAAI Conference on Artificial Intelligence_
   - [Paper](https://arxiv.org/abs/1509.06461)

### Books

1. **Sutton, R. S., & Barto, A. G. (2018)**
   - _Reinforcement Learning: An Introduction_ (2nd ed.)
   - Chapters 6 (TD Learning) and 13 (Policy Gradient Methods)
   - [Free Online](http://incompleteideas.net/book/the-book-2nd.html)

### Additional Resources

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
- [DeepMind x UCL RL Lecture Series](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)
- [Berkeley CS285 Deep RL](http://rail.eecs.berkeley.edu/deeprlcourse/)

## 💡 Discussion Questions

1. **Why does Q-Learning converge to optimal policy despite being off-policy?**
2. **In what scenarios would you prefer SARSA over Q-Learning?**

3. **How does experience replay improve sample efficiency?**

4. **Why does the max operator in Q-Learning cause overestimation?**

5. **What are the trade-offs in target network update frequency?**

## 🎓 Extensions & Challenges

### Easy

- Implement different exploration strategies (Boltzmann, UCB)
- Try different network architectures
- Visualize learned Q-functions

### Medium

- Implement Prioritized Experience Replay
- Add Dueling DQN architecture
- Compare with N-step returns

### Hard

- Implement Rainbow DQN (combines 6 extensions)
- Apply to Atari environments
- Implement distributional RL (C51, QR-DQN)

## 📝 Assignment Submission

### Required Deliverables

1. **Code:** Completed Jupyter notebooks with all cells executed
2. **Report:** Analysis of results, comparison plots, answers to questions
3. **Plots:** Learning curves, Q-value evolution, policy visualizations

### Grading Rubric

- **Implementation (40%)**: Correct and efficient code
- **Analysis (30%)**: Insightful comparison and discussion
- **Experiments (20%)**: Comprehensive evaluation with multiple runs
- **Presentation (10%)**: Clear explanations and visualizations

---

**Course:** Deep Reinforcement Learning  
**Institution:** [Your University]  
**Last Updated:** 2024

For questions or issues, please open an issue in the course repository or contact the teaching staff.
