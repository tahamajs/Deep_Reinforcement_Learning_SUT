# HW8: Exploration Methods - Solutions

This directory contains comprehensive solutions to HW8 on Exploration Methods in Reinforcement Learning.

## 📄 Files

### Main Solution Document

- **`HW8_Solutions.md`** - Complete solutions in IEEE format (~2300 lines, ~12,000 words)
  - Covers all 10 main problem sections
  - Includes theoretical analysis and practical implementations
  - Features 30+ code blocks, 100+ equations, and 15+ tables

## 📋 Contents Overview

### Section 1: Multi-Armed Bandits Fundamentals

- Formal definition of MAB problem
- Regret analysis and lower bounds
- Information-theoretic arguments

### Section 2: Epsilon-Greedy vs UCB Analysis

- Complete implementation of ε-greedy algorithm
- UCB1 algorithm with regret proof
- Comparison and empirical analysis

### Section 3: Thompson Sampling

- Bayesian approach to bandits
- Implementation for Bernoulli and Gaussian rewards
- Regret bounds and optimality

### Section 4: Count-Based Exploration

- Exploration bonuses ($\beta/\sqrt{N}$)
- R-Max algorithm implementation
- Theoretical motivation from Hoeffding bounds

### Section 5: Intrinsic Motivation Methods

- Forward model prediction error
- ICM (Intrinsic Curiosity Module)
- Solution to "Noisy TV" problem

### Section 6: Random Network Distillation (RND)

- Complete RND implementation
- Integration with PPO
- Why RND works theoretically

### Section 7: Noisy Networks

- NoisyLinear layer implementation
- Factorized Gaussian noise
- Comparison with ε-greedy

### Section 8: Bootstrap DQN

- Ensemble-based deep exploration
- Bootstrap masks for training
- Uncertainty quantification

### Section 9: Regret Analysis and Sample Complexity

- PAC framework
- Sample complexity bounds for different methods
- Lower bounds and minimax optimality

### Section 10: Empirical Comparisons

- Multi-armed bandit testbed
- Deep RL exploration comparison
- Visualizations and analysis

## 🚀 How to Use

### Reading the Solutions

1. **Sequential Reading:** Start from Section 1 and progress through all sections
2. **Topic-Specific:** Jump to specific sections using the table of contents
3. **Code Focus:** Search for ` ```python` blocks for implementations
4. **Theory Focus:** Look for sections with mathematical proofs and derivations

### Running the Code

While the main document provides implementations, to run the code:

```bash
# Create a Python file from code blocks
# Example: extract epsilon-greedy implementation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy torch gymnasium matplotlib seaborn scipy

# Run your extracted code
python epsilon_greedy.py
```

### Extracting Code Blocks

The document contains complete, runnable implementations. To extract:

1. Copy the desired code block from the markdown
2. Save to a `.py` file
3. Add necessary imports at the top
4. Run the script

## 📊 Key Results

### Multi-Armed Bandits (10 arms, 10,000 steps)

| Algorithm      | Final Regret | % Optimal | Notes                    |
| -------------- | ------------ | --------- | ------------------------ |
| ε-greedy (0.1) | 800-1000     | 85-90%    | Simple but linear regret |
| UCB1           | 30-50        | 98-99%    | Near-optimal             |
| Thompson       | 25-40        | 98-99%    | Best overall             |

### Deep RL Exploration (Sparse GridWorld)

| Method      | Success Rate | Episodes to Success |
| ----------- | ------------ | ------------------- |
| Random      | 5-10%        | ~500                |
| ε-greedy    | 40-50%       | ~200                |
| Count-Based | 80-90%       | ~50                 |
| RND         | 90-95%       | ~30                 |

## 🔍 Finding Specific Content

### Quick Search Guide

- **"Question:"** - Finds problem statements
- **"Solution:"** - Finds solution beginnings
- **"Theorem:"** - Finds theoretical results
- **"```python"** - Finds code implementations
- **"Proof:"** - Finds mathematical proofs
- **"Implementation:"** - Finds implementation details

### Topics by Keyword

- **Regret bounds:** Sections 1, 2, 9
- **Bayesian methods:** Section 3 (Thompson Sampling)
- **Intrinsic motivation:** Sections 5, 6
- **Deep RL:** Sections 6, 7, 8
- **Theory:** Sections 1, 2, 9
- **Experiments:** Sections 2, 3, 10

## 📚 References

The solutions cite 26 key papers and resources:

- **Classic Papers:** Auer et al. (UCB), Thompson (1933), Lai & Robbins
- **Modern Methods:** RND (Burda 2018), Noisy Nets (Fortunato 2018), ICM (Pathak 2017)
- **Books:** Sutton & Barto, Lattimore & Szepesvári

Full bibliography in the main document.

## 💡 Usage Tips

### For Learning

1. **First Pass:** Read through sections 1-3 for fundamentals
2. **Deep Dive:** Study proofs and derivations carefully
3. **Hands-On:** Implement code examples yourself
4. **Comparison:** Run experiments comparing different methods

### For Implementation

1. **Start Simple:** Begin with ε-greedy or UCB
2. **Add Intrinsic Motivation:** Use RND for hard exploration
3. **Monitor:** Track visitation counts and regret
4. **Tune:** Adjust hyperparameters based on problem

### For Exams/Assignments

- **Key Concepts:** Focus on sections 1, 2, 3, 9
- **Memorize:** UCB formula, regret bounds, Thompson Sampling update
- **Understand:** Why different methods work, their trade-offs
- **Practice:** Derive regret bounds, implement algorithms

## 🎯 Learning Objectives Covered

- ✅ Understanding exploration-exploitation trade-off
- ✅ Multi-armed bandit algorithms (ε-greedy, UCB, Thompson)
- ✅ Regret analysis and sample complexity
- ✅ Count-based exploration methods
- ✅ Intrinsic motivation (prediction error, RND)
- ✅ Deep RL exploration (Noisy Nets, Bootstrap DQN)
- ✅ Theoretical guarantees (PAC, minimax optimality)
- ✅ Empirical evaluation and comparison

## 🔧 Code Dependencies

Main libraries used in implementations:

```python
import numpy as np              # Numerical computations
import torch                     # Deep learning
import torch.nn as nn           # Neural networks
import torch.nn.functional as F # Activation functions
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns           # Statistical visualization
from scipy.stats import beta    # For Thompson Sampling
```

## ⚠️ Important Notes

1. **Code is Educational:** Implementations prioritize clarity over performance
2. **Hyperparameters:** May need tuning for specific problems
3. **Computational Cost:** Bootstrap DQN requires significant compute
4. **Theoretical vs Empirical:** Some theoretical bounds are loose in practice

## 📞 Support

For questions or issues:

1. Review the main solution document thoroughly
2. Check the README in the parent `HW8_Exploration_Methods/` directory
3. Consult referenced papers for deeper understanding
4. Experiment with the provided code

## 🔄 Updates

**Version 1.0** (2024)

- Initial complete solution set
- All 10 sections with implementations
- Comprehensive theory and empirics

---

**Happy Learning! 🎓**

Exploration is at the heart of reinforcement learning. Master these methods and you'll be well-equipped to tackle any RL problem!
