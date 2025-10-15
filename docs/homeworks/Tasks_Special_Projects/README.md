# Special Projects and Tasks

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Projects](https://img.shields.io/badge/Type-Projects-purple.svg)](.)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)](.)

## 📋 Overview

This directory contains special projects and advanced tasks that extend beyond regular homework assignments. These projects typically involve implementing cutting-edge techniques, reproducing research papers, or tackling open-ended challenges in deep reinforcement learning.

## 📂 Directory Structure

```
Tasks_Special_Projects/
├── code/                                    # Implementation code
├── reports/
│   ├── Task 1 Bootstrap DQN Variants.html
│   └── Task 2 Random Network Distillation.html
└── README.md
```

## 🎯 Learning Objectives

1. **Research Implementation**: Reproduce results from recent papers
2. **Advanced Techniques**: Implement state-of-the-art methods
3. **Open-Ended Problems**: Tackle challenges without predefined solutions
4. **Experimental Design**: Design and execute comprehensive experiments
5. **Technical Writing**: Document findings and insights
6. **Critical Analysis**: Evaluate strengths and limitations

## 📚 Projects

### Task 1: Bootstrap DQN Variants

**Overview:** Implement and compare bootstrapped DQN approaches for deep exploration

**Key Concepts:**

- **Bootstrap DQN**: Train ensemble of Q-networks with different bootstrapped samples
- **Deep Exploration**: Use disagreement among ensemble for exploration
- **Uncertainty Estimation**: Quantify epistemic uncertainty through ensemble variance

**Implementation Goals:**

1. Implement Bootstrap DQN with K heads
2. Implement exploration strategies:
   - Thompson sampling over Q-functions
   - UCB-style bonuses from ensemble disagreement
3. Compare with baseline DQN and other exploration methods
4. Test on environments requiring deep exploration (e.g., Chain, Deep Sea)

**Architecture:**

```python
class BootstrapDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=10):
        super().__init__()

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Multiple heads
        self.heads = nn.ModuleList([
            nn.Linear(128, action_dim)
            for _ in range(num_heads)
        ])

        self.num_heads = num_heads

    def forward(self, state, head_idx=None):
        features = self.shared(state)

        if head_idx is not None:
            # Single head
            return self.heads[head_idx](features)
        else:
            # All heads
            return torch.stack([head(features) for head in self.heads], dim=0)

    def get_uncertainty(self, state):
        """Compute uncertainty as ensemble variance"""
        q_values = self.forward(state)  # [num_heads, batch, actions]
        return q_values.var(dim=0)
```

**Key Papers:**

- Osband, I., et al. (2016) - "Deep Exploration via Bootstrapped DQN" - NIPS
- Osband, I., et al. (2018) - "Randomized Prior Functions for Deep RL" - NIPS

**Deliverables:**

- Working Bootstrap DQN implementation
- Comparison with baselines across 3+ environments
- Analysis of exploration behavior (state visitation, uncertainty estimates)
- Report documenting findings

---

### Task 2: Random Network Distillation (RND)

**Overview:** Implement RND for exploration in sparse reward environments

**Key Concepts:**

- **Random Network Distillation**: Use prediction error of random network as exploration bonus
- **Intrinsic Motivation**: Bonus rewards for novel states
- **Normalization**: Proper reward/observation normalization crucial
- **Episodic vs Non-episodic**: Different bonus computation strategies

**Implementation Goals:**

1. Implement RND with:
   - Fixed random target network
   - Trainable predictor network
   - Intrinsic reward based on prediction error
2. Implement proper normalization:
   - Running mean normalization for observations
   - Separate value functions for intrinsic/extrinsic rewards
3. Test on hard exploration games (Montezuma's Revenge, Pitfall)
4. Ablation studies on key components

**Architecture:**

```python
class RND(nn.Module):
    def __init__(self, obs_dim, feature_dim=256):
        super().__init__()

        # Fixed random target network
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # Trainable predictor network
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, obs):
        # Normalize observations
        obs = self.normalize(obs)

        # Get features
        target_features = self.target(obs)
        predicted_features = self.predictor(obs)

        # Prediction error as intrinsic reward
        intrinsic_reward = F.mse_loss(
            predicted_features,
            target_features.detach(),
            reduction='none'
        ).mean(dim=-1)

        return intrinsic_reward
```

**Key Insights:**

- RND works because frequently visited states → predictor learns well → low error → low bonus
- Novel states → high prediction error → high bonus → exploration
- Random network provides stable, non-changing target
- Proper normalization essential for stability

**Key Papers:**

- Burda, Y., et al. (2018) - "Exploration by Random Network Distillation" - ICLR
- Burda, Y., et al. (2018) - "Large-Scale Study of Curiosity-Driven Learning" - arXiv

**Deliverables:**

- Complete RND implementation with PPO
- Experiments on Atari hard exploration games
- Ablation study: importance of components (normalization, episodic bonus, etc.)
- Visualization of intrinsic rewards over time
- Report with analysis and insights

---

## 🔧 Technical Requirements

### Environment Setup

```bash
# Core libraries
pip install torch>=2.0.0
pip install gymnasium[atari]>=0.28.0
pip install ale-py>=0.8.0

# Visualization
pip install matplotlib seaborn
pip install tensorboard

# Additional tools
pip install tqdm pandas numpy
```

### Computational Resources

- **Bootstrap DQN**: Moderate (GPU recommended)
- **RND**: High (GPU required for Atari)
- **Training Time**: 1-4 hours per experiment

## 📊 Evaluation Criteria

### Code Quality (30%)

- Clean, readable, well-documented code
- Modular design with reusable components
- Proper git usage and version control
- Reproducible results (seeds, hyperparameters documented)

### Experimental Rigor (30%)

- Multiple random seeds (≥3)
- Appropriate baselines for comparison
- Statistical significance testing
- Error bars/confidence intervals on plots

### Analysis and Insights (30%)

- Deep understanding of methods
- Thoughtful ablation studies
- Discussion of failure cases
- Connection to theoretical foundations

### Presentation (10%)

- Clear visualizations
- Well-structured report
- Effective communication of results
- Professional documentation

## 📖 Additional Resources

### Papers

1. **Osband et al. (2016)** - Bootstrap DQN
2. **Burda et al. (2018)** - RND
3. **Fortunato et al. (2018)** - Noisy Networks (related exploration)
4. **Pathak et al. (2017)** - ICM (related intrinsic motivation)

### Code References

- OpenAI Baselines: https://github.com/openai/baselines
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- CleanRL: https://github.com/vwxyzjn/cleanrl

### Tutorials

- Spinning Up in Deep RL: https://spinningup.openai.com/
- Lil'Log Exploration Strategies: https://lilianweng.github.io/posts/2020-06-07-exploration-drl/

## 💡 Tips for Success

1. **Start Simple**: Implement basic version first, then add complexity
2. **Debug Thoroughly**: Test each component independently
3. **Hyperparameter Tuning**: Use small environments first (CartPole, Acrobot)
4. **Track Everything**: Use TensorBoard to monitor training
5. **Reproducibility**: Set seeds everywhere (PyTorch, NumPy, environment)
6. **Compare Carefully**: Ensure fair comparison (same hyperparameters, seeds)

## 🎓 Extension Ideas

### Bootstrap DQN Extensions

- Implement randomized prior functions
- Try different ensemble sizes
- Combine with prioritized experience replay
- Apply to continuous control

### RND Extensions

- Implement Never Give Up (NGU)
- Try episodic curiosity
- Combine with count-based exploration
- Apply to procedurally generated environments

## 📝 Submission Guidelines

1. **Code**: Well-organized repository with README
2. **Report**: PDF or HTML with:
   - Introduction and motivation
   - Methods description
   - Experimental setup
   - Results and analysis
   - Discussion and conclusion
3. **Figures**: High-quality plots with error bars
4. **Reproducibility**: Instructions to reproduce results

---

**Course:** Deep Reinforcement Learning  
**Type:** Advanced Projects  
**Difficulty:** ★★★★☆  
**Last Updated:** 2024

For questions or clarifications, contact the course staff or post in the discussion forum.
