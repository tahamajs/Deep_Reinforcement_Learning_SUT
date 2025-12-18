# CA7: Deep Q-Networks (DQN) and Value-Based Methods

## Overview

This project provides a comprehensive implementation and analysis of Deep Q-Networks (DQN) and several advanced value-based reinforcement learning methods. Building upon foundational DQN principles, this work synthesizes improvements from key research papers to enhance stability, reduce bias, and improve sample efficiency. The core algorithms implemented include:

-   **Vanilla DQN**: The foundational algorithm with experience replay and target networks.
-   **Double DQN**: Addresses overestimation bias inherent in standard DQN.
-   **Dueling DQN**: Improves Q-value estimation by decoupling state-value and advantage functions.
-   **Dueling Double DQN**: Combines the benefits of both Dueling and Double DQN.
-   **Noisy DQN**: Replaces epsilon-greedy exploration with parameter-space noise for more effective exploration.
-   **(Planned) Rainbow DQN Components**: Integration of Prioritized Experience Replay, N-step Q-learning, and Distributional RL (C51) to achieve state-of-the-art performance.

The project emphasizes a modular code structure, rigorous theoretical grounding, and comprehensive experimental analysis.

## Research Synthesis: Beyond Vanilla DQN

This project synthesizes ideas primarily from:

1.  **"Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)**: The original DQN paper, introducing the core concepts of using deep neural networks to approximate Q-functions, coupled with experience replay and target networks for stability.
2.  **"Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)**: Addresses the overestimation bias of DQN by decoupling the action selection and action evaluation processes, leading to more accurate value estimates.
3.  **"Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)**: Proposes a network architecture that explicitly separates the representation of state-value and advantage functions, improving the agent's ability to generalize across actions without affecting the policy.
4.  **"Noisy Networks for Exploration" (Fortunato et al., 2017)**: Introduces stochastic layers (noisy nets) that add noise to the network weights, allowing the agent to explore more efficiently without relying on an external epsilon-greedy schedule.
5.  **"Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017)**: A seminal work that combines six key DQN extensions (Double DQN, Prioritized Replay, Dueling Networks, Multi-step Learning, Distributional RL, and Noisy Nets) to achieve a new state-of-the-art in Atari games.

Our novel synthesis combines these elements into a structured codebase, providing both individual implementations and a pathway towards a full Rainbow DQN agent. The primary research gap addressed is the integration of these sophisticated techniques into a clean, modular, and theoretically-aligned framework suitable for educational and research purposes, allowing for easy experimentation with various combinations of these improvements.

## Project Structure

```
CA07_DQN_Value_Based_Methods/
├── README.md                    # This comprehensive guide
├── run.sh                       # Script to run all experiments
├── test_implementation.py       # Unit tests for verification
├── training_examples.py         # Example training and analysis scripts
├── requirements.txt             # Python dependencies
├── src/                         # Modularized core implementations
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Centralized hyperparameters
│   ├── agents.py                # DQN agent implementations (Vanilla, Double, Dueling, Noisy)
│   ├── models.py                # Neural network architectures (QNetwork, DuelingQNetwork, NoisyLinear, NoisyQNetwork)
│   ├── data.py                  # Data structures (ReplayBuffer, PrioritizedReplayBuffer - planned)
│   ├── utils.py                 # General utility functions (seeding, smoothing)
│   └── losses.py                # Custom loss functions (e.g., for Distributional RL - planned)
├── notebooks/                   # Jupyter notebooks for interactive walkthroughs and visualizations
│   └── main.ipynb               # Main execution and visualization notebook (planned)
├── pictures/                    # Generated plots and figures from notebooks
├── visualizations/              # Generated plots from scripts
├── results/                     # Experiment results and data
└── logs/                        # Execution logs
```

## Code Map

### `src/config.py`

-   **Purpose**: Centralizes all hyperparameters and configuration settings for the entire project. This includes environment details, agent parameters (learning rates, discount factors, buffer sizes), exploration strategies (epsilon-greedy, noisy net parameters), and advanced Rainbow DQN components (PER, N-step, C51).
-   **Key Classes**: `DQNConfig`

### `src/data.py`

-   **Purpose**: Implements data structures necessary for experience replay.
-   **Key Classes**:
    -   `ReplayBuffer`: Stores `(state, action, reward, next_state, done)` tuples for off-policy training.
    -   `(Planned) PrioritizedReplayBuffer`: An extension for prioritized sampling of experiences, giving more importance to transitions with high TD-error.

### `src/models.py`

-   **Purpose**: Defines the neural network architectures used as Q-functions.
-   **Key Classes**:
    -   `QNetwork`: A simple feed-forward neural network for basic Q-value approximation.
    -   `DuelingQNetwork`: Implements the dueling architecture, separating state-value (`V`) and advantage (`A`) streams.
    -   `NoisyLinear`: A linear layer with parameter-space noise, used in `NoisyQNetwork` for exploration.
    -   `NoisyQNetwork`: A Q-network built with `NoisyLinear` layers, replacing epsilon-greedy.
    -   `(Planned) CategoricalQNetwork`: For Distributional RL (C51), outputs a distribution over Q-values.

### `src/agents.py`

-   **Purpose**: Contains the implementations of various DQN agents. Each agent wraps a Q-network, handles experience replay, and manages the training loop logic.
-   **Key Classes**:
    -   `DQNAgent`: Basic DQN agent.
    -   `DoubleDQNAgent`: Extends `DQNAgent` with Double Q-learning updates.
    -   `DuelingDQNAgent`: Extends `DQNAgent` with a `DuelingQNetwork`.
    -   `DuelingDoubleDQNAgent`: Combines `DoubleDQNAgent` logic with a `DuelingQNetwork`.
    -   `NoisyDQNAgent`: Extends `DQNAgent` to use `NoisyQNetwork` for exploration.
    -   `(Planned) RainbowDQNAgent`: Will combine all Rainbow components into a single agent.

### `src/losses.py`

-   **Purpose**: Will contain custom loss functions, especially for advanced variants.
-   **Key Functions**:
    -   `(Planned) c51_loss`: Loss function for Distributional Reinforcement Learning (C51).

### `src/utils.py`

-   **Purpose**: Provides general utility functions for the project.
-   **Key Functions**:
    -   `set_seed`: Ensures reproducibility across random number generators.
    -   `smooth_curve`: Applies a rolling average for visualizing noisy learning curves.
    -   `(Planned) PerformanceTracker`, `ExperimentLogger`, `save_results`, `load_results`, etc.

## Installation

1.  **Clone or navigate to the project directory**:
    ```bash
    cd CAs/Solutions/CA07_DQN_Value_Based_Methods
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4.  **For GPU support (optional)**:
    If you have a CUDA-enabled GPU, ensure `torch` and `torchvision` are installed with CUDA support. Check PyTorch's official website for the correct command for your specific CUDA version. Example for CUDA 11.8:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### 1. Run All Experiments

The `run.sh` script executes a comprehensive suite of experiments, including training various DQN agents, hyperparameter studies, and robustness analyses.

```bash
chmod +x run.sh
./run.sh
```

This script will:
-   Set up the Python environment.
-   Run training for Vanilla DQN, Double DQN, Dueling DQN, Dueling Double DQN, and Noisy DQN.
-   Conduct hyperparameter optimization studies.
-   Perform robustness analyses (e.g., across random seeds and reward scales).
-   Generate plots and save results in the `visualizations/`, `results/`, and `logs/` directories.

### 2. Test Implementation

Verify the correctness of the core implementations using `pytest`:

```bash
python -m pytest test_implementation.py
```

### 3. Run Individual Experiments/Demos

You can run specific training and analysis examples:

```bash
python training_examples.py
```

Or execute individual experiment scripts (these will be updated or replaced by `training_examples.py`):

```bash
# Example (existing, may be deprecated in favor of training_examples.py)
python experiments/basic_dqn_experiment.py
```

### 4. Interactive Development with Jupyter Notebook

The `notebooks/main.ipynb` (planned) will provide an interactive environment for:
-   Exploring theoretical concepts.
-   Visualizing agent behavior and learning curves.
-   Conducting ad-hoc experiments and analysis.
-   Generating publication-quality figures for the report.

To run the notebook:
```bash
jupyter lab
# Then open notebooks/main.ipynb
```

## Theoretical Foundations and Mathematical Derivations

### Q-learning Basics

The core idea of Q-learning is to learn an action-value function \(Q(s, a)\) that estimates the expected total future reward (return) for taking action \(a\) in state \(s\) and thereafter following an optimal policy. The optimal Q-function, \(Q^*(s, a)\), satisfies the Bellman optimality equation:

\[
Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]
\]

where \(r\) is the immediate reward, \(\gamma\) is the discount factor, and \(s'\) is the next state.

### Deep Q-Networks (DQN)

DQN approximates the optimal action-value function \(Q^*(s, a)\) using a deep neural network, \(Q(s, a; \theta)\), parameterized by \(\theta\). To stabilize training, DQN employs:

1.  **Experience Replay**: Stores transitions \((s_t, a_t, r_t, s_{t+1}, d_t)\) in a replay buffer and samples mini-batches randomly. This breaks correlations between consecutive samples and improves data efficiency.
2.  **Target Network**: Uses a separate target Q-network, \(Q(s, a; \theta^-)\), with parameters \(\theta^-\) that are periodically updated from the online network's parameters \(\theta\). This creates a stable target for the Q-value updates, preventing oscillations.

The loss function for DQN is the Mean Squared Error (MSE) between the current Q-estimate and the target Q-value:

\[
L(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim U(\mathcal{D})} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
\]

where \(U(\mathcal{D})\) denotes sampling uniformly from the replay buffer \(\mathcal{D}\).

### Double DQN

DQN is known to suffer from overestimation bias due to the `max` operator in the target Q-value calculation. Double DQN addresses this by decoupling the action selection from the action evaluation. The target Q-value is calculated as:

\[
Y^{DoubleDQN}_t = r_{t+1} + \gamma Q(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta); \theta^-)
\]

Here, the online network \(\theta\) is used to select the best action \(a'\) in the next state \(s_{t+1}\), but the target network \(\theta^-\) is used to evaluate the Q-value for that action. This prevents the same network from both selecting and evaluating the action, reducing overestimation.

### Dueling DQN

Dueling DQN proposes a new network architecture rather than a change to the learning algorithm. The Q-network is decomposed into two separate streams:

1.  **State-Value Stream**: Outputs the value of the state, \(V(s; \xi)\).
2.  **Advantage Stream**: Outputs the advantage of each action, \(A(s, a; \psi)\).

These two streams are then combined to produce the Q-values:

\[
Q(s, a; \theta) = V(s; \xi) + \left( A(s, a; \psi) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \psi) \right)
\]

where \(\theta = (\xi, \psi)\) represents the combined parameters of the two streams. This decomposition allows the network to learn the value of states independently of the actions, which can be particularly beneficial in environments where many actions have similar effects on the environment.

### Noisy Networks for Exploration

Traditional DQN uses an \(\epsilon\)-greedy policy for exploration, where \(\epsilon\) decays over time. Noisy Networks introduce noise directly into the weights of the neural network, allowing the agent to learn its own exploration strategy. Each linear layer \(W\mathbf{x} + b\) is replaced by a noisy layer:

\[
(W + \sigma^w \odot \epsilon^w) \mathbf{x} + (b + \sigma^b \odot \epsilon^b)
\]

where \(\sigma^w\) and \(\sigma^b\) are learnable standard deviations, and \(\epsilon^w, \epsilon^b\) are random variables. The noise is reset at the beginning of each episode, encouraging consistent exploration over an episode.

## (Planned) Rainbow DQN Components

### Prioritized Experience Replay (PER)

Instead of uniformly sampling experiences from the replay buffer, PER samples transitions based on their temporal difference (TD) error magnitude. Transitions with higher TD error are sampled more frequently, as they are considered more "surprising" or informative. This can lead to faster learning.

The probability of sampling transition \(i\) is given by:

\[
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
\]

where \(p_i = |\delt-i| + \epsilon\) (\(\delt-i\) is the TD error, \(\epsilon\) is a small positive constant to ensure non-zero probability), and \(\alpha\) is a prioritization exponent. Importance Sampling (IS) weights are used to correct for the bias introduced by non-uniform sampling:

\[
w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta
\]

where \(N\) is the buffer size and \(\beta\) is an annealing exponent.

### N-step Q-learning

Standard DQN uses a 1-step Bellman target. N-step Q-learning extends this by considering \(N\) future rewards before bootstrapping from the target Q-network. The \(N\)-step return is calculated as:

\[
G_t^{(N)} = \sum_{k=0}^{N-1} \gamma^k r_{t+k+1} + \gamma^N \max_{a'} Q(s_{t+N}, a'; \theta^-)
\]

This provides a richer signal for learning, bridging the gap between 1-step Q-learning and Monte Carlo methods, often leading to faster and more stable learning.

### Distributional RL (C51)

Instead of learning the expected Q-value \(Q(s, a)\), Distributional RL (specifically C51, Categorical 51-atom distribution) learns a categorical distribution over possible returns. The Q-value is then the expectation of this learned distribution. The target distribution is projected onto the support of the current Q-network, and the loss minimizes the KL-divergence between the projected target distribution and the predicted distribution.

## Experimental Analysis

This section will detail the experimental setup, environments used, and expected results.

### Environments

-   **CartPole-v1**: A classic control problem, balancing a pole on a cart. Discrete action space, continuous state space.
-   **MountainCar-v0**: A sparse reward problem, pushing an underpowered car up a hill. Discrete action space, continuous state space.
-   **Acrobot-v1**: An underactuated system, swinging a two-link robot to reach a target height. Discrete action space, continuous state space.

### Planned Experiments

1.  **Comparison of DQN Variants**: Train and compare the performance (learning curves, final average rewards, training stability) of Vanilla DQN, Double DQN, Dueling DQN, Dueling Double DQN, and Noisy DQN.
2.  **Hyperparameter Sensitivity Analysis**: Investigate the impact of learning rate, hidden layer size, and exploration schedule on agent performance.
3.  **Robustness Analysis**: Evaluate agent performance across different random seeds and reward scaling factors to assess robustness.
4.  **Ablation Study of Rainbow Components**: (Planned, once implemented) Analyze the individual and combined contributions of Prioritized Replay, N-step learning, and Distributional RL.

### Expected Results

-   Double DQN is expected to outperform Vanilla DQN by reducing overestimation bias.
-   Dueling DQN should show improved sample efficiency and stability compared to Vanilla DQN, especially in environments where state values are more consistent across actions.
-   Dueling Double DQN is anticipated to combine the benefits, yielding robust performance.
-   Noisy DQN is expected to demonstrate more effective and consistent exploration than epsilon-greedy, potentially leading to faster convergence or higher final rewards.
-   Rainbow DQN (once fully implemented) should achieve superior performance across all metrics due to the synergistic combination of its components.

## Installation and Setup Notes

### Recommended Python Version

Python 3.8+ is recommended.

### Virtual Environment

Always use a virtual environment to manage dependencies.

### GPU Acceleration

Leverage CUDA-enabled GPUs for faster training. Ensure your PyTorch installation is compatible with your GPU drivers.

## Contributing

Contributions are welcome! Please ensure that any new code adheres to the project's coding standards, includes appropriate type hints and docstrings, and is covered by tests where applicable.

## References

1.  Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with Deep Reinforcement Learning. *arXiv preprint arXiv:1312.5602*.
2.  Van Hasselt, H., Guez, A., & Silver, D. (2015). Deep Reinforcement Learning with Double Q-learning. *arXiv preprint arXiv:1509.06461*.
3.  Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Silver, D., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *arXiv preprint arXiv:1511.06581*.
4.  Fortunato, M., Azar, M. G., Piot, B., Hessel, M., Budden, J., van Hasselt, H., ... & Blundell, C. (2017). Noisy Networks for Exploration. *arXiv preprint arXiv:1706.10295*.
5.  Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., ... & Silver, D. (2017). Rainbow: Combining Improvements in Deep Reinforcement Learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1).

## License

This project is part of the Deep Reinforcement Learning course materials at Sharif University of Technology.
