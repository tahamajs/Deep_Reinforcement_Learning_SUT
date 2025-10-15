# Spring 2024 Course Materials

[![Semester](https://img.shields.io/badge/Semester-Spring_2024-green.svg)](.)
[![Status](https://img.shields.io/badge/Status-Archived-gray.svg)](.)

## 📋 Overview

Spring 2024 represents an evolved version of the course with faster progression to deep RL, more implementation focus, and the addition of a project component with research posters.

## 📂 Directory Structure

```
sp24/
├── exams/
│   ├── midterm/
│   │   ├── RL_Midterm.pdf
│   │   └── SP24_RL_Midterm.zip
│   ├── final/
│   │   ├── RL_Final.pdf
│   │   └── SP24_RL_Final.zip
│   └── quizzes/
│       ├── RL_Quiz1_Key.pdf
│       ├── RL_Quiz2_Key.pdf
│       └── SP24_RL_Quizzes.zip
├── hws/
│   ├── HW1/  # Deep Q-Networks
│   ├── HW2/  # Temporal Difference & Policy Gradients
│   ├── HW3/  # On-Policy Methods (PPO)
│   └── HW4/  # Model-Based & Planning
├── posters/
│   └── [17 student research project posters]
└── README.md
```

## 🎯 Course Overview

### Instructional Philosophy

**Spring 2024 Approach:**

- **Deep RL First:** Quick introduction to neural network methods
- **Implementation Focus:** More coding, less derivation
- **Modern Techniques:** Latest algorithms and best practices
- **Project-Based Learning:** Research poster presentations

### Key Changes from SP23

| Aspect          | SP23         | SP24                   |
| --------------- | ------------ | ---------------------- |
| **Pace**        | Gradual      | Accelerated            |
| **Assignments** | 5 (HW0-4)    | 4 (HW1-4)              |
| **Quizzes**     | 6 quizzes    | 2 comprehensive        |
| **Focus**       | Theory-heavy | Implementation-focused |
| **Project**     | None         | Research posters       |
| **Framework**   | Gym 0.21     | Gymnasium 0.28+        |

## 📚 Course Content

### Module Breakdown

#### Weeks 1-4: Deep RL Fundamentals

- **Topics:** MDPs (quick review), DQN, experience replay, target networks
- **Assignment:** HW1 (Deep Q-Networks implementation)
- **Assessment:** Quiz 1

**Accelerated Start:**

- Assumes ML background
- Quick MDP review (1 week vs 3 weeks in SP23)
- Immediate neural network usage
- Modern best practices

#### Weeks 5-7: Temporal Difference & Policy Gradients

- **Topics:** TD learning, SARSA, Q-learning, REINFORCE, baselines
- **Assignment:** HW2 (TD methods & basic policy gradients)
- **Assessment:** Midterm

**Integration:**

- Combines tabular and function approximation
- Connects value-based and policy-based
- Practical focus on what works

#### Weeks 8-10: On-Policy Methods

- **Topics:** A2C, A3C, PPO, trust regions, GAE
- **Assignment:** HW3 (PPO implementation)
- **Assessment:** Quiz 2

**State-of-the-Art:**

- Focus on PPO as workhorse algorithm
- Best practices for implementation
- Hyperparameter tuning strategies
- Real-world considerations

#### Weeks 11-15: Advanced Topics & Projects

- **Topics:** Model-based RL, MCTS, MPC, exploration, multi-agent
- **Assignment:** HW4 (Model-based methods)
- **Project:** Research posters
- **Assessment:** Final exam

**Breadth:**

- Survey of advanced topics
- Student-driven projects
- Connections to current research
- Open problems and future directions

## 📝 Assignments

### HW1: Deep Q-Networks (Weeks 1-4)

**Objectives:**

- Implement DQN from scratch
- Add experience replay
- Implement target networks
- Compare DQN vs Double DQN

**Deliverables:**

- Working DQN implementation
- Experiments on CartPole and LunarLander
- Analysis of replay buffer size
- Comparison plots

**New Features vs SP23:**

- Uses Gymnasium (new Gym API)
- Modern PyTorch patterns
- Vectorized environments
- Tensorboard logging

---

### HW2: TD Methods & Policy Gradients (Weeks 5-7)

**Part 1: Temporal Difference Learning**

- Q-learning on discrete tasks
- SARSA comparison
- Off-policy evaluation

**Part 2: Policy Gradients**

- REINFORCE implementation
- Baseline addition
- Continuous action spaces

**Integration:**

- Compare value-based vs policy-based
- When to use which approach
- Sample efficiency analysis

---

### HW3: Proximal Policy Optimization (Weeks 8-10)

**Full PPO Implementation:**

- Clipped objective
- Generalized Advantage Estimation (GAE)
- Value function learning
- Multiple epochs per batch

**Continuous Control:**

- MuJoCo environments
- Hyperparameter tuning
- Ablation studies
- Performance baselines

**Key Learning:**

- State-of-the-art on-policy method
- Practical implementation details
- What makes PPO work well
- Common pitfalls and solutions

---

### HW4: Model-Based Methods (Weeks 11-13)

**Topics:**

- Learn dynamics model
- Model predictive control
- Dyna-style planning
- Compare sample efficiency

**Implementation:**

- Neural network dynamics model
- CEM or random shooting
- Integration with model-free
- Uncertainty quantification

## 🎨 Research Projects & Posters

### Overview

**New Component in SP24:** Research project with poster presentation

**Timeline:**

- Week 8: Project proposals
- Week 12: Progress presentations
- Week 15: Final posters and presentations

### Project Topics (Examples from SP24)

1. **Exploration Methods**

   - Automated Reinforcement Learning (AutoRL)
   - Curiosity-driven learning
   - RND implementation

2. **Advanced Algorithms**

   - Advancements in Offline RL
   - Safe RL methods
   - Hierarchical RL

3. **Applications**

   - Robotics manipulation
   - Game playing
   - Resource allocation

4. **Novel Architectures**
   - Attention mechanisms in RL
   - Transformer-based agents
   - Graph neural networks

### Poster Format

**Requirements:**

- Research question clearly stated
- Related work summary
- Methodology explanation
- Experimental results
- Conclusions and future work

**Evaluation Criteria:**

- Originality and depth
- Technical correctness
- Experimental rigor
- Presentation quality
- Ability to answer questions

## 📊 Assessments

### Quiz Structure (Simplified from SP23)

**Two Comprehensive Quizzes:**

**Quiz 1** (Week 4): Deep RL Foundations

- DQN architecture and training
- Experience replay benefits
- Target network purpose
- Common debugging issues

**Quiz 2** (Week 10): Policy-Based Methods

- Policy gradient theorem
- PPO objective and clipping
- Actor-critic architectures
- Practical considerations

**Format:**

- 45-60 minutes
- Mix of conceptual and practical
- Some coding questions
- Open book/notes

### Exams

**Midterm** (Week 7): Covers weeks 1-7

- Less mathematical than SP23
- More implementation-focused
- Debugging scenarios
- Algorithm selection

**Final** (Finals Week): Comprehensive

- Emphasis on weeks 8-15
- Integration of concepts
- Project-related questions
- Current research discussion

## 🔧 Technical Setup

### Required Software (Updated)

```bash
# Python 3.9+
python >= 3.9

# Core packages
gymnasium >= 0.28.0  # New Gym
numpy >= 1.21.0
torch >= 2.0.0

# Visualization
matplotlib >= 3.5.0
tensorboard >= 2.9.0

# Optional (for HW4)
mujoco >= 2.3.0
```

### Installation Instructions

```bash
# Create environment
conda create -n rl_sp24 python=3.10
conda activate rl_sp24

# Install core
pip install gymnasium[all]
pip install torch torchvision
pip install matplotlib seaborn tensorboard

# For MuJoCo environments
pip install mujoco gymnasium[mujoco]
```

## 📖 Resources

### Primary Materials

- Sutton & Barto (2018) - Still primary textbook
- Spinning Up by OpenAI - Increased emphasis
- Recent papers (2023-2024)

### New Resources

- CleanRL implementations
- Stable-Baselines3 documentation
- Hugging Face RL course
- David Silver lectures (still relevant)

## 💡 Student Feedback & Outcomes

### Positive Aspects

- Faster path to interesting applications
- More modern implementations
- Project work appreciated
- Better preparation for research

### Challenges

- Steep learning curve for those weak in ML
- Less time for theoretical depth
- Heavy workload with project
- Need strong programming skills

### Success Factors

- Prior deep learning experience
- Comfort with PyTorch
- Time management (project + 4 HWs)
- Active engagement with materials

## 🔗 Connections to SP23

### What Was Retained

- Core RL concepts
- Fundamental algorithms
- Sutton & Barto as primary text
- Mathematical foundations (condensed)

### What Was Added

- Project component
- Modern frameworks (Gymnasium)
- Recent research papers
- Best practices emphasis

### What Was Streamlined

- Fewer, longer assignments
- Fewer quizzes (but more comprehensive)
- Faster progression to deep RL
- Less time on proofs

## 📈 Poster Gallery

The `posters/` directory contains 17 student research projects, showcasing:

- Novel algorithm implementations
- Application domains
- Experimental comparisons
- Theoretical contributions

**Notable Projects:**

- Offline RL advancements
- AutoRL systems
- Safe RL methods
- Multi-agent coordination
- Exploration techniques

See `posters/README.md` for detailed descriptions.

---

**Semester:** Spring 2024  
**Instructor:** [Instructor Name]  
**Format:** Hybrid (in-person + online resources)  
**Last Updated:** Archive maintained 2024
