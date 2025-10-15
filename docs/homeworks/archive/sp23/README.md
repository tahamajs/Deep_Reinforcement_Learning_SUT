# Spring 2023 Course Materials

[![Semester](https://img.shields.io/badge/Semester-Spring_2023-blue.svg)](.)
[![Status](https://img.shields.io/badge/Status-Archived-gray.svg)](.)

## 📋 Overview

This directory contains all materials from the Spring 2023 offering of Deep Reinforcement Learning. This semester focused on building strong theoretical foundations before progressing to deep learning methods.

## 📂 Directory Structure

```
sp23/
├── exams/
│   ├── midterm/
│   │   ├── RL_Midterm.pdf
│   │   └── SP23_RL_Midterm.zip
│   ├── final/
│   │   ├── RL_Final.pdf
│   │   └── RL_Final_Solution.pdf
│   └── quizzes/
│       ├── RL_Quiz_1.pdf & Solution
│       ├── RL_Quiz_2.pdf
│       ├── RL_Quiz_4.pdf
│       ├── RL_Quiz_5.pdf
│       └── RL_Quiz_6.pdf
├── hws/
│   ├── HW0/  # Course setup and introduction
│   ├── HW1/  # Tabular RL methods
│   ├── HW2/  # Policy gradient basics
│   ├── HW3/  # Model-based and planning
│   └── HW4/  # Off-policy methods
└── README.md
```

## 🎯 Course Overview

### Instructional Philosophy

**Spring 2023 Approach:**
- **Theory First:** Strong mathematical foundations
- **Gradual Progression:** From simple to complex
- **Classical Methods:** Thorough coverage of pre-deep-learning RL
- **Rigorous Assessment:** Frequent quizzes to ensure understanding

### Target Audience

- Graduate students in CS/ML
- Strong mathematical background expected
- Previous ML experience helpful but not required
- Focus on research-oriented students

## 📚 Course Content

### Module Breakdown

#### Week 1-3: Foundations
- **Topics:** MDPs, Bellman equations, dynamic programming
- **Assignment:** HW0 (Course setup, basic concepts)
- **Assessment:** Quiz 1

**Key Concepts:**
- Markov property
- Value functions
- Policy evaluation and improvement
- Contraction mappings

#### Week 4-6: Tabular Methods
- **Topics:** Monte Carlo, TD learning, Q-learning, SARSA
- **Assignment:** HW1 (Implementation of tabular methods)
- **Assessment:** Quiz 2, Midterm

**Key Concepts:**
- Bootstrapping
- On-policy vs off-policy
- Exploration strategies
- Convergence guarantees

#### Week 7-9: Function Approximation
- **Topics:** Linear FA, gradient methods, DQN introduction
- **Assignment:** HW2 (Policy gradients, REINFORCE)
- **Assessment:** Quiz 4

**Key Concepts:**
- Generalization
- The deadly triad
- Experience replay
- Target networks

#### Week 10-12: Policy Gradients
- **Topics:** REINFORCE, actor-critic, baseline methods
- **Assignment:** HW3 (MCTS, Dyna, planning)
- **Assessment:** Quiz 5

**Key Concepts:**
- Policy parameterization
- Variance reduction
- Advantage functions
- Natural gradients

#### Week 13-15: Advanced Topics
- **Topics:** Importance sampling, off-policy evaluation
- **Assignment:** HW4 (Soft Actor-Critic, advanced methods)
- **Assessment:** Quiz 6, Final Exam

**Key Concepts:**
- Off-policy learning
- Trust regions
- Maximum entropy RL
- Practical considerations

## 📝 Assignments

### HW0: Introduction and Setup (Week 1-2)

**Objectives:**
- Set up development environment
- Understand MDP formulation
- Implement basic environment interactions
- Practice with OpenAI Gym

**Skills Developed:**
- Python programming for RL
- Environment wrappers
- Visualization techniques
- Basic debugging

**Typical Issues:**
- Environment installation
- API understanding
- Numpy operations

---

### HW1: Tabular Methods (Week 4-6)

**Part 1: CartPole with Tabular Methods**
- Discretize continuous state space
- Implement Q-learning
- Compare with SARSA

**Part 2: GridWorld Navigation**
- Implement MC prediction
- TD(0) prediction
- Compare convergence rates

**Key Learning:**
- Trade-offs between MC and TD
- Importance of exploration
- Convergence properties

---

### HW2: Policy Gradients (Week 7-9)

**Part 1: REINFORCE on CartPole**
- Implement basic REINFORCE
- Add baseline
- Analyze variance

**Part 2: Continuous Action Spaces**
- Gaussian policies
- MountainCar continuous
- Reward shaping

**Key Learning:**
- High variance in policy gradients
- Baseline importance
- Continuous action handling

---

### HW3: Model-Based Methods (Week 10-12)

**Part 1: Dyna-Q**
- Implement model learning
- Planning with model
- Compare sample efficiency

**Part 2: MCTS**
- Tree search implementation
- UCB selection
- Game playing (TicTacToe)

**Part 3: MPC**
- Learn dynamics model
- Random shooting
- CEM optimization

**Key Learning:**
- Model-based vs model-free trade-offs
- Planning efficiency
- Model errors

---

### HW4: Advanced Off-Policy Methods (Week 13-15)

**Part 1: Importance Sampling**
- Off-policy evaluation
- IS correction
- Practical challenges

**Part 2: Soft Actor-Critic**
- Implement SAC
- Automatic temperature tuning
- Compare with DDPG

**Key Learning:**
- Off-policy learning challenges
- Maximum entropy framework
- State-of-the-art continuous control

## 📊 Assessments

### Quiz Structure

**Format:**
- 20-30 minutes
- Multiple choice and short answer
- Focuses on recent material
- Closed book

**Topics by Quiz:**

1. **Quiz 1** (Week 3): MDPs, dynamic programming
2. **Quiz 2** (Week 5): MC methods, TD learning
3. **Quiz 4** (Week 8): Function approximation, DQN
4. **Quiz 5** (Week 11): Policy gradients, actor-critic
5. **Quiz 6** (Week 14): Off-policy methods, advanced topics

### Exams

**Midterm Exam** (Week 7)
- Covers weeks 1-6
- Mix of theory and application
- 2 hours
- Includes:
  - Mathematical derivations
  - Algorithm design
  - Conceptual questions
  - Short proofs

**Final Exam** (Finals Week)
- Comprehensive
- Emphasis on weeks 7-15
- 3 hours
- Includes:
  - Integration of concepts
  - Compare/contrast questions
  - Design considerations
  - Open-ended problems

## 🔧 Technical Setup

### Required Software

```bash
# Python environment
python >= 3.8
numpy >= 1.21
matplotlib >= 3.4

# RL environments
gym == 0.21.0  # Note: Using older Gym, not Gymnasium
```

### Installation (SP23 Version)

```bash
# Create environment
conda create -n rl_sp23 python=3.8
conda activate rl_sp23

# Install dependencies
pip install gym==0.21.0
pip install numpy matplotlib torch
```

### Known Issues

- **Gym Deprecation:** SP23 used old Gym API
- **API Changes:** Some methods renamed in newer versions
- **Compatibility:** Code may need updates for current environments

## 📖 Textbook and References

### Primary Textbook

**Sutton & Barto (2018)** - Reinforcement Learning: An Introduction

**Chapter Mapping:**
- Weeks 1-3: Chapters 3-4 (MDPs, DP)
- Weeks 4-6: Chapters 5-6 (MC, TD)
- Weeks 7-9: Chapters 9-10 (Function Approximation)
- Weeks 10-12: Chapters 13 (Policy Gradient)
- Weeks 13-15: Chapters 12 (Eligibility Traces, Off-policy)

### Supplementary Materials

- David Silver's RL Course (Video lectures)
- Berkeley CS285 lecture notes
- OpenAI Spinning Up guide

## 💡 Study Tips for SP23 Materials

### Understanding Theory

1. **Work Through Derivations:** Don't just memorize
2. **Visualize Concepts:** Draw value functions, policy updates
3. **Connect to Code:** Link theory to implementation
4. **Practice Proofs:** Especially for convergence guarantees

### Implementation Strategy

1. **Start Simple:** Test on toy problems first
2. **Debug Systematically:** Print intermediate values
3. **Compare Results:** Use provided baselines
4. **Understand Hyperparameters:** Don't just tune randomly

### Exam Preparation

1. **Review All Quizzes:** Common question patterns
2. **Redo Homework:** Focus on conceptual parts
3. **Study Groups:** Explain concepts to others
4. **Past Materials:** If available from previous years

## 📈 Typical Student Outcomes

### Strengths Developed

- Strong theoretical understanding
- Ability to derive and prove
- Mathematical maturity in RL
- Classical algorithm implementation

### Common Challenges

- High mathematical rigor
- Time-consuming homeworks
- Frequent assessments (6 quizzes)
- Gap between theory and practice

### Success Factors

- Consistent weekly effort
- Active participation
- Strong math background
- Peer collaboration

## 🔗 Connections to Current Course

### What Carried Forward

- Core MDP theory
- Fundamental algorithms
- Mathematical foundations
- Implementation practices

### What Changed in SP24

- Faster progression to deep RL
- Fewer but longer assignments
- More modern implementations
- Added project component

## ⚠️ Using These Materials

### For Self-Study

**Advantages:**
- Thorough theoretical coverage
- Well-structured progression
- Comprehensive assessments

**Considerations:**
- Time-intensive
- May need to update code
- Heavy on mathematics

### For Comparison

- See how topics evolved
- Compare homework difficulty
- Study different explanations
- Review alternative examples

---

**Semester:** Spring 2023  
**Instructor:** [Instructor Name]  
**Format:** In-person lectures, online resources  
**Last Updated:** Archive maintained 2024

