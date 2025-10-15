# Weekly Course Materials

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Materials](https://img.shields.io/badge/Type-Course--Materials-blue.svg)](.)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)](.)

## 📋 Overview

This directory contains weekly course materials including lecture summaries, reading assignments, discussion topics, and supplementary resources. Materials are organized by week and correspond to the course schedule.

## 📂 Directory Structure

```
Weekly_Materials/
├── resources/
│   ├── week1.md       # Week 1: Introduction to RL
│   ├── week2.md       # Week 2: Multi-Armed Bandits
│   ├── week3.md       # Week 3: MDPs and Dynamic Programming
│   ├── week4.md       # Week 4: Model-Free Prediction
│   ├── week5.md       # Week 5: Model-Free Control
│   ├── week6.md       # Week 6: Value Function Approximation
│   ├── week7.md       # Week 7: Deep Q-Networks
│   ├── week8.md       # Week 8: Policy Gradient Methods
│   ├── week9.md       # Week 9: Actor-Critic Methods
│   ├── week10.md      # Week 10: Model-Based RL
│   ├── week11.md      # Week 11: Exploration & Intrinsic Motivation
│   ├── week12.md      # Week 12: Multi-Agent RL
│   ├── week13.md      # Week 13: Meta-Learning & Transfer
│   └── week14.md      # Week 14: Advanced Topics & Future Directions
└── README.md
```

## 📅 Course Schedule

### Module 1: Foundations (Weeks 1-3)

**Week 1: Introduction to Reinforcement Learning**

- RL problem formulation
- Exploration vs exploitation
- Key challenges and applications
- Historical context

**Week 2: Multi-Armed Bandits**

- Stateless RL
- Regret minimization
- Exploration strategies (ε-greedy, UCB, Thompson Sampling)
- Contextual bandits

**Week 3: Markov Decision Processes**

- MDPs formulation
- Value functions and Bellman equations
- Dynamic programming (policy iteration, value iteration)
- Optimality and convergence

### Module 2: Tabular Methods (Weeks 4-5)

**Week 4: Model-Free Prediction**

- Monte Carlo methods
- Temporal Difference learning
- TD(λ) and eligibility traces
- Bias-variance trade-offs

**Week 5: Model-Free Control**

- SARSA (on-policy)
- Q-Learning (off-policy)
- Expected SARSA
- Convergence properties

### Module 3: Function Approximation (Weeks 6-7)

**Week 6: Value Function Approximation**

- Linear function approximation
- Feature engineering
- Gradient descent methods
- The deadly triad

**Week 7: Deep Q-Networks**

- Neural network function approximators
- Experience replay
- Target networks
- DQN variants (Double DQN, Dueling DQN, Rainbow)

### Module 4: Policy-Based Methods (Weeks 8-9)

**Week 8: Policy Gradient Methods**

- Policy gradient theorem
- REINFORCE algorithm
- Baseline methods
- Variance reduction techniques

**Week 9: Actor-Critic Methods**

- Advantage actor-critic (A2C)
- Asynchronous advantage actor-critic (A3C)
- Proximal policy optimization (PPO)
- Trust region methods (TRPO)

### Module 5: Advanced Topics (Weeks 10-14)

**Week 10: Model-Based RL**

- World models
- Planning with learned models
- Dyna architecture
- Model predictive control

**Week 11: Exploration & Intrinsic Motivation**

- Count-based exploration
- Curiosity-driven learning
- Random network distillation
- Never Give Up (NGU)

**Week 12: Multi-Agent RL**

- Game theory foundations
- Multi-agent learning algorithms
- Cooperation and competition
- Communication and coordination

**Week 13: Meta-Learning & Transfer**

- Learning to learn
- MAML for RL
- Task distributions
- Few-shot adaptation

**Week 14: Advanced Topics**

- Offline RL
- Safe RL
- Hierarchical RL
- Current research frontiers

## 📖 How to Use These Materials

### Before Class

1. Read the assigned materials for the week
2. Review previous week's concepts
3. Prepare questions on unclear topics
4. Complete any pre-class quizzes

### During Class

1. Take notes on key concepts
2. Participate in discussions
3. Ask clarifying questions
4. Work on in-class exercises

### After Class

1. Review lecture notes and materials
2. Complete homework assignments
3. Read supplementary papers
4. Practice implementing algorithms

## 🎯 Learning Resources

### Textbooks

1. **Sutton & Barto (2018)** - _Reinforcement Learning: An Introduction_ (2nd ed.)

   - Primary textbook for the course
   - [Free Online](http://incompleteideas.net/book/the-book-2nd.html)

2. **Szepesvári (2010)** - _Algorithms for Reinforcement Learning_
   - Concise mathematical treatment
   - [Free Online](https://sites.ualberta.ca/~szepesva/RLBook.html)

### Video Lectures

1. **David Silver's RL Course** (DeepMind/UCL)
   - [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
2. **Berkeley CS285: Deep RL** (Sergey Levine)
   - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
3. **DeepMind x UCL RL Lecture Series**
   - [Course Website](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)

### Online Resources

1. **OpenAI Spinning Up**

   - Excellent practical introduction
   - [Website](https://spinningup.openai.com/)

2. **Lil'Log**

   - Detailed blog posts on RL topics
   - [Blog](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)

3. **Distill.pub**
   - Visual explanations of RL concepts
   - [Website](https://distill.pub/)

### Code Repositories

1. **Stable-Baselines3**

   - Reliable implementations of RL algorithms
   - [GitHub](https://github.com/DLR-RM/stable-baselines3)

2. **CleanRL**

   - Single-file implementations for learning
   - [GitHub](https://github.com/vwxyzjn/cleanrl)

3. **RLlib**
   - Scalable RL library
   - [Documentation](https://docs.ray.io/en/latest/rllib/index.html)

## 📝 Weekly Structure

Each week typically includes:

### Lecture Components

- **Concepts**: Core ideas and theory
- **Algorithms**: Step-by-step procedures
- **Examples**: Concrete applications
- **Code**: Implementation demonstrations

### Assignments

- **Reading**: Papers and textbook chapters
- **Problem Sets**: Mathematical exercises
- **Programming**: Implementation tasks
- **Discussion**: Forum participation

### Assessment

- **Quizzes**: Check understanding
- **Homework**: Apply concepts
- **Projects**: Open-ended tasks
- **Exams**: Comprehensive evaluation

## 🔧 Prerequisites

### Mathematics

- **Linear Algebra**: Vectors, matrices, eigenvalues
- **Calculus**: Derivatives, gradients, chain rule
- **Probability**: Expectations, distributions, Bayes' rule
- **Optimization**: Gradient descent, convexity

### Programming

- **Python**: Numpy, PyTorch/TensorFlow
- **Git**: Version control basics
- **Jupyter**: Notebook usage
- **Debugging**: Testing and troubleshooting

### Machine Learning

- **Supervised Learning**: Classification, regression
- **Neural Networks**: Backpropagation, architectures
- **Optimization**: SGD, Adam, learning rates
- **Generalization**: Overfitting, regularization

## 💡 Study Tips

### For Theory

1. Work through proofs step-by-step
2. Understand intuition before formalism
3. Create concept maps connecting ideas
4. Practice deriving key equations

### For Implementation

1. Start with simple environments
2. Debug incrementally
3. Compare with reference implementations
4. Visualize learned policies and values

### For Exams

1. Review weekly summaries
2. Practice past problems
3. Form study groups
4. Focus on understanding, not memorization

## 🎓 Additional Topics

### Research Seminars

- Guest lectures from researchers
- Paper reading groups
- Recent developments in RL
- Career paths in RL

### Office Hours

- Professor: [Schedule TBD]
- TAs: [Schedule TBD]
- Format: In-person / Virtual

### Discussion Forum

- Ask questions
- Help classmates
- Share resources
- Organize study groups

## 📊 Grading Breakdown

| Component               | Weight |
| ----------------------- | ------ |
| Homework Assignments    | 40%    |
| Midterm Exam            | 20%    |
| Final Project           | 25%    |
| Quizzes & Participation | 10%    |
| Final Exam              | 5%     |

## 🔗 Related Courses

### Prerequisites

- **CS/MATH**: Probability, Linear Algebra, Optimization
- **CS**: Machine Learning, Neural Networks
- **CS**: Algorithms, Data Structures

### Follow-up Courses

- **Advanced Topics in RL**
- **Multi-Agent Systems**
- **Robotics and Control**
- **Neuroscience and AI**

## 📅 Important Dates

- **Week 1**: Course introduction
- **Week 7**: Midterm exam
- **Week 10**: Project proposal due
- **Week 14**: Final presentations
- **Finals Week**: Final exam & project submission

## 🆘 Getting Help

### When Stuck

1. Review lecture materials and textbook
2. Search documentation and tutorials
3. Ask on discussion forum
4. Attend office hours
5. Form study groups

### Common Issues

- **Math**: Review prerequisites, seek tutoring
- **Coding**: Debug systematically, use print statements
- **Concepts**: Draw diagrams, create examples
- **Time Management**: Start early, break down tasks

---

**Course:** Deep Reinforcement Learning  
**Semester:** [Current Semester]  
**Instructor:** [Instructor Name]  
**Last Updated:** 2024

For course updates, check the main course website or learning management system.
