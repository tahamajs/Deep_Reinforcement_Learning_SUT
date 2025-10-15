# Spring 2023 Exams and Quizzes

[![Exams](https://img.shields.io/badge/Type-Assessments-red.svg)](.)
[![SP23](https://img.shields.io/badge/Semester-Spring_2023-blue.svg)](.)

## 📋 Overview

This directory contains all exam materials from Spring 2023, including the midterm, final exam, and six quizzes. These assessments emphasize theoretical understanding and mathematical rigor.

## 📂 Contents

```
exams/
├── midterm/
│   ├── RL_Midterm.pdf              # Midterm exam questions
│   └── SP23_RL_Midterm.zip         # Archive with supporting files
├── final/
│   ├── RL_Final.pdf                # Final exam questions
│   └── RL_Final_Solution.pdf       # Final exam solutions
└── quizzes/
    ├── RL_Quiz_1.pdf               # Week 3: MDPs & DP
    ├── RL_Quiz_1_Solution.pdf
    ├── RL_Quiz_2.pdf               # Week 5: MC & TD
    ├── RL_Quiz_4.pdf               # Week 8: Function Approximation
    ├── RL_Quiz_5.pdf               # Week 11: Policy Gradients
    └── RL_Quiz_6.pdf               # Week 14: Off-Policy Methods
```

## 📝 Midterm Exam

**Week:** 7 (Mid-semester)  
**Duration:** 120 minutes  
**Format:** Closed book, formula sheet allowed  
**Coverage:** Weeks 1-6

### Topics Covered

#### Core Concepts (40%)
- Markov Decision Processes
- Bellman equations
- Dynamic programming algorithms
- Policy and value iteration

#### Tabular Methods (35%)
- Monte Carlo methods
- Temporal Difference learning
- Q-learning and SARSA
- Exploration strategies

#### Theoretical Questions (25%)
- Convergence proofs
- Derivations
- Comparison of algorithms
- Complexity analysis

### Question Types

1. **Definitions and Concepts** (20 points)
   - Define key terms
   - Explain relationships
   - Provide intuitions

2. **Mathematical Derivations** (30 points)
   - Derive Bellman equations
   - Prove convergence properties
   - Show optimality conditions

3. **Algorithm Design** (25 points)
   - Implement algorithms in pseudocode
   - Analyze time complexity
   - Modify for specific scenarios

4. **Applied Problems** (25 points)
   - Solve small MDPs by hand
   - Trace algorithm execution
   - Compute value functions

### Study Recommendations

- Review Sutton & Barto Chapters 3-6
- Redo all homework problems
- Work through Quiz 1 and Quiz 2
- Practice manual value iteration
- Understand all derivations from lectures

## 📄 Final Exam

**Week:** Finals Week  
**Duration:** 180 minutes  
**Format:** Closed book, two-page formula sheet allowed  
**Coverage:** Comprehensive (emphasis on weeks 7-15)

### Topics Covered

#### Deep Reinforcement Learning (30%)
- Function approximation
- Deep Q-Networks
- Experience replay
- Target networks

#### Policy Gradient Methods (30%)
- REINFORCE algorithm
- Actor-critic architectures
- Variance reduction techniques
- Natural policy gradients

#### Advanced Topics (25%)
- Model-based RL
- Off-policy learning
- Importance sampling
- Trust region methods

#### Integration and Comparison (15%)
- Compare algorithm families
- Design considerations
- Practical applications
- Current research directions

### Question Types

1. **Conceptual Understanding** (35 points)
   - Explain core concepts deeply
   - Compare and contrast methods
   - Discuss trade-offs

2. **Mathematical Analysis** (40 points)
   - Derive gradient updates
   - Analyze convergence
   - Prove properties

3. **Algorithm Implementation** (30 points)
   - Write detailed pseudocode
   - Explain design choices
   - Handle edge cases

4. **Open-Ended Design** (20 points)
   - Propose solutions to novel problems
   - Justify architectural choices
   - Discuss potential issues

5. **Synthesis Questions** (25 points)
   - Integrate multiple concepts
   - Apply to new scenarios
   - Theoretical-practical connections

### Study Recommendations

- Comprehensive review of all lectures
- Redo all homework assignments
- Review all 6 quizzes
- Study final solutions when available
- Form study groups for discussion

## 📊 Quizzes

**Format:** 20-30 minutes, in-class  
**Frequency:** Approximately every 2-3 weeks  
**Weight:** 15% of final grade (combined)

### Quiz 1: MDPs and Dynamic Programming

**Week:** 3  
**Topics:**
- MDP formulation (states, actions, rewards, transitions)
- Bellman expectation and optimality equations
- Policy evaluation
- Policy iteration
- Value iteration
- Asynchronous DP

**Sample Questions:**
- Define the Bellman optimality equation for V*
- Given small MDP, compute optimal policy by hand
- Prove policy improvement theorem
- Compare policy vs value iteration

**Key Skills:**
- Write Bellman equations
- Compute value functions manually
- Understand optimality conditions
- Analyze convergence

---

### Quiz 2: Monte Carlo and TD Methods

**Week:** 5  
**Topics:**
- Monte Carlo prediction
- Monte Carlo control
- TD(0) prediction
- Q-learning
- SARSA
- Expected SARSA

**Sample Questions:**
- Explain bias-variance trade-off MC vs TD
- Derive Q-learning update rule
- Why is Q-learning off-policy?
- Given trajectory, show TD updates

**Key Skills:**
- Distinguish MC from TD
- Understand bootstrapping
- Know on-policy vs off-policy
- Trace algorithm execution

---

### Quiz 4: Function Approximation

**Week:** 8  
**Topics:**
- Linear function approximation
- Gradient descent methods
- Semi-gradient methods
- The deadly triad
- Deep Q-Networks
- Experience replay
- Target networks

**Sample Questions:**
- What is the deadly triad and why is it problematic?
- Explain how experience replay helps DQN
- Derive semi-gradient TD update
- Why use target networks?

**Key Skills:**
- Understand function approximation challenges
- Know DQN innovations
- Explain stability techniques
- Analyze convergence issues

---

### Quiz 5: Policy Gradient Methods

**Week:** 11  
**Topics:**
- Policy gradient theorem
- REINFORCE algorithm
- Baseline methods
- Actor-critic architecture
- Advantage functions
- Natural gradients

**Sample Questions:**
- State and explain policy gradient theorem
- Why do policy gradients have high variance?
- How do baselines reduce variance without bias?
- Compare value-based vs policy-based methods

**Key Skills:**
- Derive policy gradients
- Understand variance reduction
- Know actor-critic synergy
- Analyze convergence properties

---

### Quiz 6: Off-Policy and Advanced Methods

**Week:** 14  
**Topics:**
- Importance sampling
- Off-policy evaluation
- Trust region methods
- Maximum entropy RL
- Soft Actor-Critic
- Practical considerations

**Sample Questions:**
- Derive importance sampling ratio
- Explain trust region constraint in TRPO
- What is maximum entropy objective in SAC?
- Compare on-policy vs off-policy sample efficiency

**Key Skills:**
- Understand off-policy corrections
- Know trust region motivation
- Explain entropy regularization
- Analyze practical trade-offs

## 📈 Grading Rubric

### Midterm Breakdown
- **Correctness** (60%): Accurate answers and derivations
- **Clarity** (20%): Clear explanations and notation
- **Completeness** (15%): Address all parts of questions
- **Insight** (5%): Demonstrate deep understanding

### Final Exam Breakdown
- **Technical Accuracy** (50%): Correct solutions
- **Conceptual Understanding** (25%): Deep comprehension
- **Communication** (15%): Clear presentation
- **Creativity** (10%): Novel insights and connections

### Quiz Grading
- **Accuracy** (70%): Correct answers
- **Reasoning** (20%): Logical explanations
- **Presentation** (10%): Neat and organized

## 💡 Exam Preparation Tips

### General Strategy
1. **Start Early:** Don't cram
2. **Active Review:** Work problems, don't just read
3. **Study Groups:** Explain to others
4. **Past Materials:** Use old quizzes as practice
5. **Office Hours:** Ask clarifying questions

### For Theoretical Questions
- Understand derivations, don't memorize
- Practice writing proofs
- Know all assumptions in theorems
- Connect math to intuition

### For Applied Questions
- Practice on toy problems
- Trace algorithms by hand
- Understand when algorithms fail
- Know practical considerations

### Common Pitfalls
- Confusing on-policy vs off-policy
- Mixing up different value functions
- Forgetting discount factors in calculations
- Not checking boundary conditions

## 📚 Additional Study Resources

### Recommended Review
- Sutton & Barto: All chapters covered in class
- Lecture slides: Focus on derivations
- Homework solutions: Understand every step
- Quiz solutions: Study error patterns

### Practice Problems
- Textbook exercises
- Past exam problems (if available)
- Online problem sets
- Create your own examples

### Key Equations to Know
- Bellman equations (expectation and optimality)
- TD error and updates
- Policy gradient theorem
- Q-learning vs SARSA updates
- DQN loss function
- REINFORCE gradient

---

**Note:** These exams test deep understanding, not just memorization. Focus on concepts, derivations, and intuitions.

**Last Updated:** 2024 (Archive)

