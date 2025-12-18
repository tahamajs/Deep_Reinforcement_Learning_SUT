# CA5: Advanced Deep Q-Network (DQN) Methods - Comprehensive Implementation

## Abstract

This project provides a comprehensive, modular implementation of advanced Deep Q-Network (DQN) methods. It addresses the challenges of stability and performance in deep reinforcement learning by incorporating key architectural and algorithmic improvements: Double DQN for mitigating overestimation bias, Dueling DQN for enhanced state-value representation, and Prioritized Experience Replay for improved sample efficiency. The codebase is designed with a strong emphasis on clean, type-hinted Python modules, supported by a Jupyter notebook for interactive experimentation and visualization. This document serves as a detailed guide, covering the theoretical foundations, mathematical derivations, architectural design, and practical usage of each component.

**Index Terms** — Deep Q-Networks, Reinforcement Learning, Double DQN, Dueling DQN, Prioritized Experience Replay, Value-Based Methods, Deep Learning.

## 1. Introduction: Advancing Deep Q-Learning

Deep Reinforcement Learning (DRL), particularly value-based methods like Deep Q-Networks (DQN), has achieved remarkable success in various domains. However, standard DQN suffers from several limitations, including overestimation of Q-values, instability during training, and inefficient use of experience. This project explores and implements state-of-the-art enhancements that address these issues, leading to more robust and performant agents.

The core objective of this assignment is to:
1.  **Modularize** the implementation of various DQN agents and their foundational components (replay buffers, network architectures).
2.  **Elucidate** the theoretical underpinnings and mathematical derivations of each advanced technique.
3.  **Provide** a clean, well-documented, and type-hinted Python codebase.
4.  **Demonstrate** the comparative performance and characteristics of these agents through systematic experiments and visualizations.

## 2. Theoretical Framework and Methodology

This section details the theoretical background and mathematical formulations for each advanced DQN method implemented.

### 2.1. Deep Q-Networks (DQN) - The Foundation

DQN combines Q-learning with deep neural networks. The agent learns an approximation of the optimal action-value function, \(Q^*(s, a)\), which represents the maximum expected return achievable by starting in state \(s\), taking action \(a\), and then following the optimal policy thereafter.

**Bellman Equation for Optimal Q-Function:**
\[
Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{P}(s'|s,a)} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]
\]
where \(r\) is the immediate reward, \(\gamma\) is the discount factor, and \(s'\) is the next state.

**DQN Loss Function (Mean Squared Bellman Error - MSBE):**
To train the Q-network, we minimize the MSBE between the current Q-value and the target Q-value:
\[
L(w) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ \left( y_t - Q(s, a; w) \right)^2 \right]
\]
where \(\mathcal{D}\) is the experience replay buffer, \(w\) are the parameters of the online Q-network, and \(y_t\) is the target Q-value.

**Target Q-value for Vanilla DQN:**
\[
y_t = r + \gamma (1 - d) \max_{a'} Q(s', a'; w^-)
\]
Here, \(w^-\) are the parameters of a separate **target network**, which is updated periodically from the online network's weights \(w\) to stabilize training. \(d\) is a binary flag indicating if \(s'\) is a terminal state (1 for terminal, 0 otherwise).

### 2.2. Double Deep Q-Networks (Double DQN)

Vanilla DQN often suffers from overestimation of Q-values, leading to suboptimal policies. Double DQN addresses this by decoupling the action selection from the action evaluation.

**Double DQN Target Q-value:**
\[
y_t = r + \gamma (1 - d) Q(s', \arg\max_{a'} Q(s', a'; w); w^-)
\]
In this formulation:
- The **online network** (\(Q(s', \cdot; w)\)) selects the greedy action \(a'\) for the next state \(s'\).
- The **target network** (\(Q(s', \cdot; w^-)\)) evaluates the Q-value for that selected action \(a'\).

This separation helps to reduce the positive bias in Q-value estimation.

### 2.3. Dueling Deep Q-Networks (Dueling DQN)

Dueling DQN introduces a novel network architecture that explicitly separates the estimation of state-value and advantage functions. This allows the network to learn which states are valuable independently of the actions taken, which can be particularly beneficial in environments where many actions do not affect the environment in a meaningful way.

**Q-value Decomposition:**
\[
Q(s, a; w, \alpha, \beta) = V(s; w, \alpha) + A(s, a; w, \beta)
\]
where \(V(s)\) is the state-value function and \(A(s, a)\) is the advantage function. \(\alpha\) and \(\beta\) are parameters for the value and advantage streams, respectively, which share a common convolutional feature learner parameterized by \(w\).

To ensure identifiability and prevent the value function from subsuming the advantage function, the advantage function is typically normalized:
\[
Q(s, a; w, \alpha, \beta) = V(s; w, \alpha) + \left( A(s, a; w, \beta) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; w, \beta) \right)
\]
This normalization ensures that the advantages are effectively relative to the mean advantage across all actions for a given state.

### 2.4. Prioritized Experience Replay (PER)

Standard experience replay samples transitions uniformly from the buffer. PER improves sample efficiency by prioritizing "important" transitions more frequently. Importance is typically measured by the magnitude of the Temporal Difference (TD) error, as large TD errors indicate that the agent learned something new or unexpected from that experience.

**Probability of Sampling a Transition \(i\):**
\[
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
\]
where \(p_i\) is the priority of transition \(i\), and \(\alpha\) is a hyperparameter (0 to 1) that determines how much prioritization is used. A value of \(\alpha = 0\) corresponds to uniform sampling.

**Priority Update:**
After an update, the priority \(p_i\) for a sampled transition \(i\) is set to its absolute TD error:
\[
p_i = |y_t - Q(s, a; w)|
\]
A small constant \(\epsilon\) is often added to the TD error to ensure all transitions have a non-zero probability of being sampled.

**Importance Sampling (IS) Weights:**
Prioritized sampling introduces a bias, as samples with higher TD errors are seen more frequently. To correct this bias, importance sampling weights are used during learning:
\[
IS\_weight_i = \left( \frac{1}{N P(i)} \right)^\beta
\]
where \(N\) is the number of transitions in the buffer, and \(\beta\) is a hyperparameter (0 to 1) that compensates for the bias. \(\beta\) is typically annealed linearly from an initial value (e.g., 0.4) to 1.0 during training. The weights are then normalized by \(\max_i IS\_weight_i\).

The loss function with IS weights becomes:
\[
L(w) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} \left[ IS\_weight_i \cdot \left( y_t - Q(s, a; w) \right)^2 \right]
\]

## 3. Project Structure and Codebase Overview

The codebase is organized into modular Python packages to promote clarity, reusability, and maintainability.

```
CA05_Advanced_DQN_Methods/
├── agents/                 # Implementations of various DQN agent algorithms
│   ├── __init__.py         # Package initialization
│   ├── dqn_base.py         # Base DQNAgent class with common functionalities
│   ├── double_dqn.py       # Double DQN agent implementation
│   ├── dueling_dqn.py      # Dueling DQN agent implementation
│   ├── prioritized_replay_dqn.py # DQN agent with Prioritized Experience Replay
│   └── rainbow_dqn.py      # Placeholder for Rainbow DQN (future work)
├── environments/           # Custom Gymnasium environments and wrappers
│   ├── __init__.py         # Package initialization
│   ├── complex_envs.py     # Definitions for more complex custom environments
│   └── custom_envs.py      # Definitions for simpler custom environments
├── utils/                  # Utility functions and helper classes
│   ├── __init__.py         # Package initialization
│   ├── replay_buffers.py   # Implementations of ReplayBuffer and PrioritizedReplayBuffer
│   ├── network_architectures.py # Implementations of QNetwork and DuelingQNetwork
│   ├── training_analysis.py # Tools for analyzing training metrics (e.g., smoothed rewards)
│   ├── analysis_tools.py   # General analysis and plotting utilities
│   ├── ca5_helpers.py      # Miscellaneous helper functions specific to CA5
│   └── ca5_main.py         # Older main execution logic (to be deprecated/integrated)
├── experiments/            # Experiment configurations and runners
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Centralized dataclasses for AgentConfig and ExperimentConfig
│   └── complex_experiments.py # Older experiment runner logic (to be deprecated/integrated)
├── evaluation/             # Performance evaluation utilities
│   └── __init__.py         # Package initialization (contains PerformanceEvaluator, compare_agents)
├── visualizations/         # Directory to save generated plots and figures
├── results/                # Directory to save experiment results (JSON, etc.)
├── CA5.ipynb               # Jupyter notebook for interactive examples, training, and visualization
├── main.py                 # Main Python script for running experiments from command line
├── run.sh                  # Shell script to execute full project workflow (setup, train, compare)
├── requirements.txt        # Python package dependencies
└── README.md               # This documentation file
```

## 4. Implementation Details

### 4.1. `agents/` Package

-   **`dqn_base.py`**:
    -   `DQNAgent` class: Serves as the base class for all DQN variants. It encapsulates common functionalities such as:
        -   Initialization of online and target Q-networks (`QNetwork`).
        -   Adam optimizer setup.
        -   `ReplayBuffer` management.
        -   Epsilon-greedy action selection logic.
        -   Methods for saving and loading agent states.
        -   An abstract `update()` method that must be implemented by subclasses.
        -   A `_common_update_step` method handles the basic loss calculation and target network updates, which can be reused or extended by subclasses.
-   **`double_dqn.py`**:
    -   `DoubleDQNAgent` class: Inherits from `DQNAgent`. Its `update()` method implements the Double DQN target calculation, where the online network selects the action and the target network evaluates it.
-   **`dueling_dqn.py`**:
    -   `DuelingDQNAgent` class: Inherits from `DQNAgent`. It overrides the Q-network initialization to use `DuelingQNetwork` instead of `QNetwork`. The `update()` method remains largely similar to Vanilla DQN, but leverages the Dueling architecture's outputs.
-   **`prioritized_replay_dqn.py`**:
    -   `PrioritizedDQNAgent` class: Inherits from `DQNAgent`. It replaces the standard `ReplayBuffer` with a `PrioritizedReplayBuffer`. Its `update()` method samples transitions based on priorities, calculates TD errors, applies importance sampling weights to the loss, and updates priorities in the buffer.

### 4.2. `utils/` Package

-   **`replay_buffers.py`**:
    -   `Transition` namedtuple: Defines the structure for storing experience tuples (`state`, `action`, `reward`, `next_state`, `done`).
    -   `ReplayBuffer` class: Implements a standard experience replay buffer using `collections.deque`.
    -   `PrioritizedReplayBuffer` class: Implements a prioritized experience replay buffer using a sum tree-like structure for efficient sampling and priority updates based on TD errors. It also handles importance sampling weight calculation.
-   **`network_architectures.py`**:
    -   `QNetwork` class: A simple feedforward neural network serving as the basic Q-function approximator.
    -   `DuelingQNetwork` class: Implements the Dueling architecture, separating feature extraction, state-value stream, and advantage stream, then combining them.

### 4.3. `experiments/` Package

-   **`config.py`**:
    -   `AgentConfig` dataclass: Centralizes all hyperparameters for a single DQN agent (learning rate, discount factor, epsilon schedule, buffer size, etc.), including specific parameters for PER (alpha, beta).
    -   `ExperimentConfig` dataclass: Holds configuration for a full training experiment (environment name, number of episodes, seed, paths for results/plots).
    -   `get_dqn_configs` function: Provides predefined `AgentConfig` objects for different DQN variants, allowing for easy selection and comparison.

### 4.4. `training_examples.py`

This module contains the primary functions for training agents and running comparisons:
-   `train_dqn_agent()`: Orchestrates the training loop for a single DQN agent type, logging metrics such as episode rewards, lengths, and losses.
-   `dqn_variant_comparison()`: Runs multiple trials for different DQN variants, collects their performance metrics, and generates comparative plots (e.g., average rewards, learning curves, conceptual characteristics).
-   `plot_q_value_landscape()`: Visualizes the Q-value landscape by sampling states and plotting Q-value distributions for different actions.
-   `plot_experience_replay_analysis()`: Analyzes and visualizes the contents of the replay buffer, showing distributions of rewards, actions, and state features.

### 4.5. `main.py`

The main entry point for running experiments from the command line. It uses `argparse` to handle different execution modes (`train`, `compare`, `all`), environment selection, agent type, number of episodes, and output directories. It leverages the `ExperimentConfig` and `AgentConfig` for consistent parameter management.

### 4.6. `CA5.ipynb`

A Jupyter notebook providing an interactive walkthrough of the implemented methods. It includes:
-   Initialization of the environment and agents.
-   Demonstrations of single-agent training.
-   Comparative analysis across different DQN variants.
-   Interactive plotting and visualization of Q-values, training curves, and replay buffer dynamics.

## 5. Installation and Setup

### Prerequisites

-   Python 3.8+
-   PyTorch 1.9+
-   Gymnasium (or Gym for older versions)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd CA05_Advanced_DQN_Methods
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## 6. Usage

### 6.1. Running from `main.py` (Command Line)

The `main.py` script provides a flexible interface for running training and comparisons.

**Basic Training of a Vanilla DQN Agent:**
```bash
python main.py --mode train --agent dqn --env CartPole-v1 --episodes 500
```

**Training a Dueling DQN Agent:**
```bash
python main.py --mode train --agent dueling_dqn --env LunarLander-v2 --episodes 1000 --seed 123
```

**Comparing All Implemented DQN Variants:**
```bash
python main.py --mode compare --env CartPole-v1 --episodes 500
```

**Running All Modes (Train a single agent, then run comparison):**
```bash
python main.py --mode all --agent prioritized_dqn --env CartPole-v1 --episodes 500
```
*(Note: When `--mode all` is used, the `--agent` argument specifies which single agent to train in the `train` step, while `compare` will run all configured agents.)*

### 6.2. Interactive Usage with `CA5.ipynb`

Open the `CA5.ipynb` Jupyter notebook in your environment (e.g., VS Code Jupyter extension, Jupyter Lab, or Jupyter Notebook). The notebook contains executable cells that demonstrate:
-   Environment and agent initialization.
-   Step-by-step training of individual DQN agents.
-   Comparative analysis across different DQN variants.
-   Interactive plotting and visualization of Q-values, training curves, and replay buffer dynamics.

Execute the cells sequentially to follow the comprehensive analysis.

## 7. Results and Outputs

Upon successful execution, results and visualizations will be saved in the respective directories:

-   **`visualizations/`**: Contains generated plots (e.g., `CartPole-v1_dqn_training_analysis.png`, `CartPole-v1_comparison_comparison.png`).
-   **`results/`**: Contains JSON files summarizing training runs and comparison outcomes (e.g., `dqn_CartPole-v1_training_results.json`, `CartPole-v1_comparison_results.json`, `summary_report.json`).

## 8. Conclusion and Future Work

This project has demonstrated a robust, modular implementation of advanced DQN methods. By carefully integrating Double DQN, Dueling DQN, and Prioritized Experience Replay, we can significantly improve the performance and stability of deep Q-learning agents across various environments.

**Future Work includes:**
-   Implementing **Rainbow DQN**: Combining all advanced techniques (Double DQN, Dueling DQN, PER, Multi-step Learning, Distributional RL, Noisy Nets) into a single agent.
-   Exploring **Multi-step Learning** for faster credit assignment.
-   Integrating **Distributional RL** (e.g., C51, QR-DQN) to learn the full distribution of returns.
-   Adding **Noisy Nets** for more efficient exploration.
-   Developing more sophisticated **Hyperparameter Optimization** techniques.
-   Extending to **Continuous Control Environments** with appropriate value-based adaptations (e.g., DDPG, TD3, SAC for actor-critic methods).

## 9. License

This project is released under the MIT License. See the `LICENSE` file for details.

## 10. Contact

For any questions or support, please open an issue in the GitHub repository or contact the DRL Course Team.
