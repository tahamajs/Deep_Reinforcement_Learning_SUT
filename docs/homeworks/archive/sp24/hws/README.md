# Spring 2024 - Reinforcement Learning Homeworks

This directory contains all homework assignments from the Spring 2024 semester of the Deep Reinforcement Learning course.

## 📋 Overview

The Spring 2024 homeworks reflect an updated curriculum with refined assignments focusing on core deep RL algorithms and practical applications. These assignments build progressively from value-based methods to advanced policy gradient techniques.

## 📂 Directory Structure

```
sp24/hws/
├── HW1/          # Introduction & Tabular Methods
├── HW2/          # Value-Based Deep RL
├── HW3/          # On-Policy Methods
└── HW4/          # Practical RL Applications
```

## 📚 Homework Assignments

### HW1: Introduction to Reinforcement Learning
**Topics:** MDP Fundamentals, Tabular RL Methods

**Contents:**
- `RL_HW1.pdf` - Problem set and assignment description
- `RL_HW1_Solution.pdf` - Complete solutions
- `SP24_RL_HW1.zip` - Assignment package
- `SP24_RL_HW1_Solution.zip` - Solution package

**Key Topics:**
- Markov Decision Processes (MDPs)
- Bellman Equations and Optimality
- Value Iteration and Policy Iteration
- Temporal Difference Learning
- Q-Learning and SARSA
- Tabular methods on GridWorld

**Learning Objectives:**
- Understand MDP formulation
- Implement dynamic programming methods
- Master TD learning algorithms
- Analyze convergence properties

---

### HW2: Value-Based Deep Reinforcement Learning
**Topics:** Monte Carlo, TD Learning, DQN, Double DQN

**Contents:**
- `RL_HW2.pdf` - Assignment description
- `RL_HW2_MC_TD.ipynb` - Monte Carlo and TD methods
- `RL_HW2_DQN.ipynb` - Deep Q-Network implementation
- `RL_HW2_Solution.pdf` - Complete solutions
- `SP24_RL_HW2.zip` - Assignment package

**Part 1: Monte Carlo and TD Methods**
- First-Visit Monte Carlo
- TD(0) and TD(λ)
- N-step TD methods
- Eligibility traces
- Comparison on simple environments

**Part 2: Deep Q-Networks**
- **DQN Implementation:**
  - Neural network Q-function approximation
  - Experience replay buffer
  - Target network stabilization
  - ε-greedy exploration
  
- **Double DQN:**
  - Addressing overestimation bias
  - Decoupled action selection and evaluation
  - Performance comparison

**Environments:**
- CliffWalking (tabular)
- CartPole-v1 (DQN)
- LunarLander-v2 (optional)

**Learning Objectives:**
- Understand MC vs TD trade-offs
- Implement function approximation
- Master experience replay
- Mitigate Q-value overestimation
- Analyze learning curves

---

### HW3: On-Policy Methods
**Topics:** Policy Gradients, REINFORCE, A2C, PPO

**Contents:**
- `RL_HW3.pdf` - Assignment description
- `RL_HW3_On_Policy.ipynb` - Policy gradient implementations
- `SP24_RL_HW3.zip` - Assignment package

**Implementations:**

1. **REINFORCE Algorithm:**
   - Basic policy gradient
   - Monte Carlo returns
   - Baseline for variance reduction

2. **Advantage Actor-Critic (A2C):**
   - Simultaneous policy and value learning
   - Advantage function estimation
   - Synchronous updates

3. **Proximal Policy Optimization (PPO):**
   - Clipped surrogate objective
   - Generalized Advantage Estimation (GAE)
   - Trust region approximation
   - Multiple epochs on same data

**Environments:**
- CartPole-v1
- LunarLander-v2
- Continuous control (optional)

**Comparison:**
- Sample efficiency
- Stability across runs
- Computational cost
- Hyperparameter sensitivity

**Learning Objectives:**
- Master policy gradient theorem
- Implement actor-critic methods
- Understand variance reduction techniques
- Apply trust region methods
- Compare on-policy algorithms

---

### HW4: Practical Reinforcement Learning
**Topics:** Real-World Applications, Advanced Techniques

**Contents:**
- `RL_HW4.pdf` - Assignment description
- `RL_HW4_Practical.ipynb` - Practical implementations
- `SP24_RL_HW4.zip` - Assignment package

**Project Components:**

1. **Hyperparameter Tuning:**
   - Learning rate schedules
   - Exploration strategies
   - Network architectures
   - Batch sizes and buffer sizes

2. **Algorithm Comparison:**
   - DQN vs PPO vs SAC
   - On-policy vs off-policy
   - Sample efficiency analysis
   - Computational requirements

3. **Practical Challenges:**
   - Training stability
   - Reward shaping
   - State preprocessing
   - Action space design

4. **Real-World Application:**
   - Custom environment setup
   - Problem formulation as MDP
   - Algorithm selection and justification
   - Results and analysis

**Learning Objectives:**
- Apply RL to practical problems
- Understand hyperparameter impact
- Diagnose training issues
- Compare algorithms systematically
- Build end-to-end RL solutions

---

## 🚀 Getting Started

### Environment Setup

```bash
# Create virtual environment
python -m venv rl_sp24
source rl_sp24/bin/activate  # On Windows: rl_sp24\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install gymnasium[all]
pip install torch torchvision
pip install numpy matplotlib pandas
pip install jupyter notebook
pip install stable-baselines3  # For baselines
pip install tensorboard  # For logging
```

### Running Assignments

1. **Download and extract:**
   ```bash
   unzip SP24_RL_HW2.zip
   cd SP24_RL_HW2/
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open the notebook** and follow instructions

### Recommended Workflow

1. Read the PDF assignment thoroughly
2. Review relevant course materials
3. Implement step-by-step
4. Test each component individually
5. Run full experiments
6. Analyze and visualize results
7. Answer written questions

## 📊 Grading Breakdown

Typical assignment structure:

| Component | Weight | Description |
|-----------|--------|-------------|
| Implementation | 40-50% | Correct, efficient code |
| Experiments | 20-30% | Comprehensive evaluation |
| Analysis | 15-25% | Insights and discussion |
| Written Questions | 10-15% | Theoretical understanding |
| Code Quality | 5-10% | Style, documentation, reproducibility |

## 🔧 Common Issues & Fixes

### Installation Issues

**Problem:** Gymnasium installation fails
```bash
# Solution
pip install setuptools wheel
pip install gymnasium
```

**Problem:** PyTorch CUDA not detected
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Training Issues

**Problem:** Agent not learning
- Check reward scaling
- Verify gradient flow
- Tune learning rate
- Increase training steps
- Check environment setup

**Problem:** Training unstable
- Reduce learning rate
- Increase batch size
- Use gradient clipping
- Add entropy regularization
- Check hyperparameters

### Performance Issues

**Problem:** Training too slow
- Use GPU if available
- Vectorize operations
- Reduce logging frequency
- Optimize data loading
- Profile code

## 📖 Resources

### Required Reading
1. **Sutton & Barto (2018)** - Reinforcement Learning: An Introduction
   - Chapters 1-13 (core material)
   - [Free PDF](http://incompleteideas.net/book/the-book-2nd.html)

### Papers by Assignment

**HW1:**
- Watkins & Dayan (1992) - Q-Learning

**HW2:**
- Mnih et al. (2015) - DQN
- van Hasselt et al. (2016) - Double DQN

**HW3:**
- Williams (1992) - REINFORCE
- Schulman et al. (2017) - PPO
- Mnih et al. (2016) - A3C

**HW4:**
- Haarnoja et al. (2018) - SAC
- Henderson et al. (2018) - Deep RL Reproducibility

### Online Resources
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## 💡 Best Practices

### Code Organization
```python
# Recommended structure
project/
├── agents/          # Agent implementations
├── envs/            # Custom environments
├── utils/           # Helper functions
├── configs/         # Hyperparameters
├── results/         # Saved results
└── notebooks/       # Jupyter notebooks
```

### Debugging Tips
1. **Start Simple**: Test on CartPole first
2. **Visualize**: Plot learning curves continuously
3. **Check Gradients**: Monitor gradient magnitudes
4. **Baseline**: Compare with known implementations
5. **Reproduce**: Use fixed random seeds

### Experiment Tracking
```python
# Use TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# Log metrics
writer.add_scalar('reward/episode', reward, episode)
writer.add_scalar('loss/value', loss, step)
```

## 🎯 Learning Outcomes

Upon completing these assignments, you will be able to:

✅ **Foundational Skills:**
- Formulate problems as MDPs
- Implement tabular RL algorithms
- Understand convergence guarantees

✅ **Deep RL Skills:**
- Build neural network function approximators
- Implement experience replay and target networks
- Handle high-dimensional state spaces

✅ **Advanced Skills:**
- Implement policy gradient methods
- Use advantage estimation and baselines
- Apply trust region methods

✅ **Practical Skills:**
- Debug and tune RL agents
- Compare algorithms systematically
- Build end-to-end RL applications

✅ **Research Skills:**
- Read and understand RL papers
- Reproduce published results
- Analyze algorithm behavior

## 📝 Submission Guidelines

### What to Submit
1. **Notebooks**: All cells executed with outputs
2. **Code**: Clean, documented, reproducible
3. **Report**: Analysis and answers to questions
4. **Plots**: Learning curves and comparisons
5. **README**: How to run your code

### Code Quality
- Use meaningful variable names
- Add comments for complex logic
- Follow PEP 8 style guide
- Include docstrings
- Remove debug code

### Reproducibility
```python
# Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## 🏆 Challenge Problems

For extra practice:
1. Implement Rainbow DQN (combines 6 extensions)
2. Apply PPO to Atari games
3. Build a custom environment
4. Reproduce a recent RL paper
5. Compare 5+ algorithms on same task

## 📧 Support

- **Office Hours**: Check course schedule
- **Discussion Forum**: Post questions online
- **Email**: Contact TAs for specific issues
- **Study Groups**: Form groups with classmates

---

**Course:** Deep Reinforcement Learning  
**Semester:** Spring 2024  
**Instructor:** [Course Instructor]  
**Last Updated:** 2024

---

## 📜 Academic Integrity

These materials are archived for educational purposes. If you are currently taking this course:

⚠️ **Do not copy solutions**  
✅ **Use as learning reference**  
✅ **Understand the concepts**  
✅ **Write your own code**  
✅ **Cite when appropriate**

Violating academic integrity policies can result in serious consequences.

---

For current course information and materials, visit the main course website.

