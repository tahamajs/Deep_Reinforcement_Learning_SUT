# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 1: Imitation Learning - Behavior Cloning and DAgger

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment introduces **imitation learning**, a paradigm in reinforcement learning where an agent learns to mimic expert behavior from demonstrations. You will implement Behavior Cloning (BC) and Dataset Aggregation (DAgger), comparing their performance and understanding the limitations of supervised learning approaches in sequential decision-making tasks.

Unlike traditional RL where agents learn through trial and error, imitation learning leverages pre-collected expert demonstrations to bootstrap learning, making it particularly useful when:
- Reward functions are difficult to specify
- Exploration is dangerous or expensive
- Expert demonstrations are readily available

## Learning Objectives

- Understand the fundamentals of imitation learning and its relationship to supervised learning
- Implement Behavior Cloning using supervised learning on expert trajectories
- Implement DAgger to address compounding errors and distribution shift in BC
- Analyze the differences between offline and online imitation learning
- Evaluate policies in continuous control MuJoCo environments
- Understand when and why imitation learning methods succeed or fail

## Key Concepts Explained

### 1. Behavior Cloning (BC)
**Behavior Cloning** is the simplest form of imitation learning. It treats the problem as supervised learning:
- **Input**: States (observations) from expert trajectories
- **Output**: Actions taken by the expert
- **Training**: Minimize prediction error using standard supervised learning (e.g., MSE for continuous actions, cross-entropy for discrete)

**Mathematical Formulation**:
```
π_θ(a|s) ≈ π_expert(a|s)
Loss = E[(π_θ(s) - a_expert)²]
```

**Advantages**:
- Simple to implement
- No environment interaction needed during training
- Fast training with offline data

**Limitations**:
- **Distribution shift**: The learned policy may visit states not in the training data
- **Compounding errors**: Small errors accumulate over time in sequential decision making
- **No correction mechanism**: Cannot recover from mistakes

### 2. Dataset Aggregation (DAgger)
**DAgger** addresses BC's distribution shift problem through iterative data collection:

**Algorithm**:
1. Train initial policy π₁ on expert data D₁
2. For iteration i:
   - Execute policy πᵢ to collect trajectories
   - Query expert for optimal actions on these new states
   - Aggregate: Dᵢ₊₁ = Dᵢ ∪ {new state-action pairs}
   - Train πᵢ₊₁ on Dᵢ₊₁

**Key Insight**: By collecting data from the learner's own state distribution, DAgger ensures the policy trains on states it will actually encounter.

**Advantages over BC**:
- Reduces distribution shift
- Better generalization to novel states
- Improved long-horizon performance

**Requirements**:
- Access to expert during training (online expert queries)
- More training time and environment interactions

### 3. Compounding Errors
In sequential decision-making, even small errors compound exponentially:
- If error probability per step is ε
- After T steps, expected deviation grows as O(εT²)
- This makes behavior cloning particularly challenging for long horizons

**Example**: A self-driving car that slightly drifts off-center will encounter increasingly unfamiliar states, leading to larger errors and eventual failure.

### 4. Distribution Shift
**Distribution Shift** occurs when:
- **Training distribution**: States visited by the expert p_expert(s)
- **Test distribution**: States visited by the learned policy p_π(s)
- When p_expert(s) ≠ p_π(s), the policy encounters states it wasn't trained on

DAgger addresses this by training on p_π(s) instead of p_expert(s).

### 5. Expert Demonstrations
Pre-collected trajectories from an optimal or near-optimal policy:
- Usually obtained from human experts or trained RL agents
- Quality of demonstrations directly impacts learned policy performance
- This assignment uses pre-trained MuJoCo policies as experts

## Technical Implementation Details

### Files to Complete

1. **`cs285/agents/bc_agent.py`**:
   - Implement the BC agent's training loop
   - Add supervised learning updates
   - Implement action prediction

2. **`cs285/policies/MLP_policy.py`**:
   - Neural network policy architecture
   - Forward pass for action selection
   - Backward pass for gradient updates

### Environment Setup

This homework uses **MuJoCo** (Multi-Joint dynamics with Contact) for continuous control:
- **Ant-v4**: 8-DOF quadruped robot
- **HalfCheetah-v4**: 6-DOF planar running robot  
- **Hopper-v4**: 3-DOF one-legged hopping robot
- **Walker2d-v4**: 6-DOF bipedal walker
- **Humanoid-v4**: 17-DOF humanoid (advanced)

### Neural Network Architecture

Typical policy network:
```
Input: state observation (varies by environment)
Hidden layers: 2 layers × 64 units (ReLU activation)
Output: action means (continuous) + log_std (for stochastic policies)
```

### Training Process

**Behavior Cloning**:
1. Load expert demonstrations from `.pkl` files
2. Extract (state, action) pairs
3. Train policy network via supervised learning
4. Evaluate on environment

**DAgger**:
1. Warm-start with BC on expert data
2. For n iterations:
   - Roll out current policy in environment
   - Query expert for actions on visited states
   - Add to dataset
   - Retrain policy

## Structure

- `cs285/`: Core RL codebase (agents, policies, infrastructure)
- `cs285/expert_data/`: Pre-collected expert demonstrations
- `cs285/policies/experts/`: Trained expert policy networks
- `cs285/scripts/`: Training scripts
- `README.md`: This file
- `run_hw1.sh`: Automated execution script

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
3. (Optional) Use a virtual environment for isolation.

## Running Experiments

### Manual Execution

**Behavior Cloning Example**:
```bash
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --env_name Ant-v4 \
    --exp_name bc_ant \
    --n_iter 1 \
    --batch_size 1000 \
    --eval_batch_size 1000
```

**DAgger Example**:
```bash
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --env_name Ant-v4 \
    --exp_name dagger_ant \
    --n_iter 10 \
    --do_dagger \
    --batch_size 1000 \
    --eval_batch_size 1000
```

## One-click runner script

To automate environment setup and execute the default Behavior Cloning and DAgger experiments on MuJoCo tasks, use the helper script:

```bash
chmod +x run_hw1.sh
./run_hw1.sh
```

The script will create (or reuse) a virtual environment in `.venv/`, install dependencies from `requirements.txt`, register the `cs285` package in editable mode, and then launch both experiments for `HalfCheetah-v4`. Set the following environment variables to customize the run:

- `ENV_NAME` – choose among `Ant-v4`, `HalfCheetah-v4`, `Hopper-v4`, `Walker2d-v4`, or `Humanoid-v4` (requires you to supply the missing expert dataset file).
- `RUN_BEHAVIOR_CLONING` / `RUN_DAGGER` – set to `0` to skip either phase.
- `PYTHON_BIN` – specify a different Python executable (defaults to `python3`).
- `SKIP_INSTALL` – set to `1` to reuse existing dependencies.

Example: run only DAgger for Hopper with an existing environment.

```bash
SKIP_INSTALL=1 RUN_BEHAVIOR_CLONING=0 ENV_NAME=Hopper-v4 ./run_hw1.sh
```

## Key Files

- `cs285/agents/`: RL agent implementations
- `cs285/policies/`: Policy networks
- `cs285/critics/`: Value function approximators
- `cs285/infrastructure/`: Utilities and trainers

## Submission

- Submit your code, results, and a report (if required) as specified in the assignment PDF.

## References

- Ross, S., & Bagnell, J. A. (2010). Efficient reductions for imitation learning. AISTATS.
- Pomerleau, D. A. (1988). ALVINN: An autonomous land vehicle in a neural network. NIPS.
- Sutton & Barto, "Reinforcement Learning: An Introduction"
- OpenAI Gymnasium documentation
