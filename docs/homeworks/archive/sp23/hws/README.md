# Spring 2023 Homework Assignments

[![Homework](https://img.shields.io/badge/Type-Assignments-green.svg)](.)
[![SP23](https://img.shields.io/badge/Semester-Spring_2023-blue.svg)](.)

## 📋 Overview

This directory contains five homework assignments from Spring 2023, providing a comprehensive introduction to reinforcement learning from basic concepts through advanced methods.

## 📂 Contents

```
hws/
├── HW0/                    # Introduction and Setup
│   ├── RL_HW0.pdf
│   └── SP23_RL_HW0.zip
├── HW1/                    # Tabular RL Methods
│   ├── SP23_RL_HW1/
│   ├── SP23_RL_HW1_Solutions/
│   └── Archives (.zip files)
├── HW2/                    # Policy Gradient Methods
│   ├── SP23_RL_HW2/
│   ├── RL_HW2_Solution.pdf
│   └── Archives
├── HW3/                    # Model-Based RL
│   ├── SP23_RL_HW3/
│   ├── RL_HW3_solution.pdf
│   └── Archives
└── HW4/                    # Advanced Methods
    ├── SP23_RL_HW4/
    └── Archives
```

## 📚 Assignment Details

### HW0: Introduction and Setup
**Due:** Week 2  
**Weight:** 5%  
**Difficulty:** ⭐☆☆☆☆

**Objectives:**
- Set up Python environment for RL
- Understand MDP formulation
- Practice with OpenAI Gym
- Implement basic environment interactions

**Contents:**
- Installation guide
- MDP exercises
- Simple CartPole interaction
- Visualization basics

**Skills:**
- Python setup and package management
- Gym API usage
- Basic numpy operations
- Plotting with matplotlib

---

### HW1: Tabular RL Methods
**Due:** Week 6  
**Weight:** 20%  
**Difficulty:** ⭐⭐⭐☆☆

**Topics:**
- Monte Carlo methods
- Temporal Difference learning
- Q-learning implementation
- SARSA implementation

**Notebooks:**
1. **RL_HW1_Tabular.ipynb**
   - Implement MC prediction
   - Implement TD(0) prediction
   - Compare convergence rates
   - GridWorld experiments

2. **RL_HW1_CartPole.ipynb**
   - Discretize continuous states
   - Implement Q-learning
   - Implement SARSA
   - Compare on-policy vs off-policy

**Key Concepts:**
- Bootstrapping vs Monte Carlo
- On-policy vs off-policy learning
- Exploration strategies
- Convergence properties

**Common Challenges:**
- State space discretization
- Choosing learning rates
- Balancing exploration/exploitation
- Debugging value function updates

---

### HW2: Policy Gradient Methods
**Due:** Week 9  
**Weight:** 20%  
**Difficulty:** ⭐⭐⭐⭐☆

**Topics:**
- REINFORCE algorithm
- Baseline methods
- Continuous action spaces
- Variance reduction

**Notebook:**
**RL_HW2_PPO_vs_DDPG.ipynb**

**Part 1: REINFORCE Implementation**
- Implement basic REINFORCE
- Add value function baseline
- Analyze variance reduction
- CartPole experiments

**Part 2: Continuous Control**
- Implement Gaussian policy
- MountainCar Continuous
- Compare with DQN
- Hyperparameter tuning

**Key Concepts:**
- Policy gradient theorem
- High variance problem
- Baseline benefits
- Continuous action handling

**Common Challenges:**
- Numerical stability in log probabilities
- Choosing baseline architecture
- Reward normalization
- Slow convergence

---

### HW3: Model-Based RL & Planning
**Due:** Week 12  
**Weight:** 25%  
**Difficulty:** ⭐⭐⭐⭐⭐

**Topics:**
- Dyna-Q architecture
- Monte Carlo Tree Search
- Model Predictive Control
- Planning with learned models

**Notebooks:**
1. **RL_HW3_MCTS.ipynb**
   - Implement MCTS with UCB
   - TicTacToe or Connect4
   - Analyze tree growth
   - Compare with minimax

2. **RL_HW3_Thompson_Sampling.ipynb**
   - Bayesian exploration
   - Multi-armed bandits
   - Thompson sampling implementation
   - Compare with UCB

**Key Concepts:**
- Model learning vs model-free
- Planning efficiency
- Exploration in tree search
- Sample complexity trade-offs

**Common Challenges:**
- Model accuracy and bias
- Computational cost of planning
- Tree memory management
- Balancing model usage

---

### HW4: Advanced Off-Policy Methods
**Due:** Week 15  
**Weight:** 30%  
**Difficulty:** ⭐⭐⭐⭐⭐

**Topics:**
- Soft Actor-Critic (SAC)
- Off-policy evaluation
- Maximum entropy RL
- State-of-the-art continuous control

**Notebook:**
**RL_HW4_Soft_Actor_Critic.ipynb**

**Implementation:**
- SAC algorithm from scratch
- Twin Q-networks
- Automatic temperature tuning
- Continuous control tasks

**Analysis:**
- Compare with DDPG and PPO
- Sample efficiency study
- Exploration behavior analysis
- Hyperparameter sensitivity

**Key Concepts:**
- Maximum entropy framework
- Off-policy actor-critic
- Clipped double Q-learning
- Entropy regularization

**Common Challenges:**
- Numerical stability
- Tuning multiple networks
- Replay buffer management
- Debugging complex interactions

## 🎯 Learning Progression

### Conceptual Flow

```
HW0: Basics
  ↓
HW1: Tabular Methods (exact solutions)
  ↓
HW2: Function Approximation (policy-based)
  ↓
HW3: Model-Based (planning and exploration)
  ↓
HW4: State-of-the-Art (modern deep RL)
```

### Skill Development

**Programming Skills:**
- Week 2: Basic Python, Gym API
- Week 6: Algorithm implementation, debugging
- Week 9: Neural networks, PyTorch
- Week 12: Complex algorithms, tree structures
- Week 15: Production-quality code, experiments

**Theoretical Understanding:**
- Gradual introduction of concepts
- Each assignment builds on previous
- Theory-practice connection emphasized
- Mathematical rigor increases

## 💻 Technical Requirements

### Environment Setup

```bash
# Create conda environment
conda create -n rl_sp23 python=3.8
conda activate rl_sp23

# Install core packages
pip install gym==0.21.0  # Note: Old Gym version
pip install numpy matplotlib
pip install torch torchvision

# Additional packages
pip install pandas seaborn tqdm
```

### Dependencies by Assignment

**HW0-HW1:**
- Python 3.8+
- NumPy
- Matplotlib
- Gym 0.21

**HW2-HW4:**
- Above +
- PyTorch 1.x
- Seaborn (visualization)
- Jupyter notebook

## 📊 Grading Rubric

### Code Quality (30%)
- Correctness of implementation
- Code organization and readability
- Comments and documentation
- Efficient algorithms

### Experimental Results (35%)
- Successful training
- Convergence achieved
- Hyperparameter exploration
- Multiple runs with error bars

### Analysis and Insights (25%)
- Understanding demonstrated
- Thoughtful discussion
- Comparison of methods
- Connection to theory

### Presentation (10%)
- Clear plots and visualizations
- Well-structured report
- Professional documentation
- Reproducible results

## 🐛 Common Issues and Solutions

### HW1 Issues
**Problem:** Q-learning not converging
**Solution:** Check learning rate, exploration schedule, discretization

**Problem:** SARSA too conservative
**Solution:** Tune exploration parameter, increase episodes

### HW2 Issues
**Problem:** REINFORCE high variance
**Solution:** Add baseline, normalize returns, use more episodes

**Problem:** NaN in training
**Solution:** Add epsilon to log, clip gradients, check reward scale

### HW3 Issues
**Problem:** MCTS too slow
**Solution:** Optimize tree storage, prune branches, parallelize

**Problem:** Learned model inaccurate
**Solution:** More training data, better architecture, regularization

### HW4 Issues
**Problem:** SAC not learning
**Solution:** Check all three networks updating, tune learning rates

**Problem:** Entropy collapsing
**Solution:** Verify automatic temperature working, check target entropy

## 📖 Study Resources

### Per Assignment

**HW0:** Sutton & Barto Chapter 1-3  
**HW1:** Sutton & Barto Chapter 5-6  
**HW2:** Sutton & Barto Chapter 13  
**HW3:** Sutton & Barto Chapter 8, MCTS survey papers  
**HW4:** SAC paper, Spinning Up SAC documentation

### Additional Materials
- Lecture slides for each topic
- Office hour recordings
- Discussion forum Q&A
- Example solutions (partial)

## ⏰ Time Estimates

**HW0:** 3-5 hours  
**HW1:** 10-15 hours  
**HW2:** 15-20 hours  
**HW3:** 20-25 hours  
**HW4:** 25-30 hours

**Total:** ~75-95 hours for all assignments

## 🎓 Tips for Success

1. **Start Early:** Especially HW3 and HW4
2. **Test Incrementally:** Don't implement everything at once
3. **Use Provided Code:** Build on templates when available
4. **Debug Systematically:** Print intermediate values
5. **Visualize Everything:** Plot value functions, policies, losses
6. **Compare with Baselines:** Use reference implementations
7. **Document Assumptions:** Explain design choices
8. **Ask Questions:** Use office hours and forums

---

**Academic Integrity:** These are archived materials for reference only. Do not submit old solutions as your own work.

**Last Updated:** 2024 (Archive)
