# CA6: Policy Gradient Methods - Modular Implementation and Analysis

## Overview

This directory now contains a completely refactored and modularized implementation of various policy gradient methods for deep reinforcement learning. The original monolithic Jupyter notebook and `training_examples.py` have been broken down into organized Python modules under `src/` for enhanced maintainability, reusability, and clarity. This structure aligns with best practices for building robust and scalable research codebases.

## Quick Start

To run all algorithms, generate results, and visualize performance, use the main Python script or the newly created Jupyter notebook:

```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Run all algorithms and generate results using the main script
python main.py

# Run in quick mode with fewer episodes for faster execution
python main.py --quick

# Or, open and run the modularized notebook:
jupyter notebook notebooks/main.ipynb
```

## Installation

To set up the environment, navigate to the assignment directory and install the required dependencies:

```bash
# Navigate to the assignment directory
cd CAs/Solutions/CA06_Policy_Gradient_Modular

# Install dependencies
pip install -r requirements.txt

# (Optional) Make run script executable, though main.py is now preferred
chmod +x run.sh
```

## Modular Structure

This project adheres to a strict modular structure, with all core components centralized under the `src/` directory. This design separates concerns, making it easier to understand, test, and extend the codebase.

```
CA06_Policy_Gradient_Modular/
├── main.py                     # Main execution script, orchestrates training and analysis
├── run.sh                      # (Legacy) Bash execution script, now points to main.py
├── requirements.txt            # Python dependencies
├── README.md                   # This comprehensive documentation
├── CA6.ipynb                   # (Legacy) Original Jupyter notebook (now deprecated)
├── training_examples.py        # (Legacy) Old monolithic training examples (now deprecated)
├──
├── src/                        # Core modular implementation
│   ├── config.py               # Global configuration and hyperparameters
│   ├── model.py                # Neural network architectures (Policy, Value Networks)
│   ├── agents.py               # Implementations of various policy gradient agents
│   ├── losses.py               # (Placeholder) Custom loss function definitions
│   ├── data.py                 # (Placeholder) Environment and data handling
│   └── utils.py                # Training loops, analysis functions, and plotting utilities
├──
├── notebooks/                  # Jupyter notebooks for execution and visualization
│   └── main.ipynb              # The primary notebook for running experiments and viewing results
├──
├── visualizations/             # Directory for saving generated plots and figures
├── results/                    # Directory for saving training logs and detailed results
├── logs/                       # (Optional) For more detailed system logs
└── CA6_files/                  # (Legacy) Original notebook outputs
```

### Core `src/` Modules Explained

1.  **`src/config.py`**: This file centralizes all global configurations, hyperparameters (e.g., learning rates, discount factors, episode counts), random seeds, and device settings (CPU/GPU). This ensures consistency across all agents and experiments and simplifies hyperparameter tuning.
2.  **`src/model.py`**: Contains the PyTorch `nn.Module` definitions for all neural networks used in the policy gradient agents. This includes `PolicyNetwork` (for discrete action spaces), `ValueNetwork` (for baselines and critics), and `ContinuousPolicyNetwork` (for continuous action spaces using Gaussian distributions).
3.  **`src/agents.py`**: Houses the implementations of the various reinforcement learning agents. Each agent (e.g., `REINFORCEAgent`, `REINFORCEBaselineAgent`, `ActorCriticAgent`, `PPOAgent`, `ContinuousPPOAgent`) is a self-contained class, encapsulating its policy, value networks, optimizers, and update logic.
4.  **`src/losses.py`**: Currently, simple loss functions (e.g., MSE for value, policy loss based on advantage) are integrated directly into the agent update methods. For more complex scenarios or custom loss formulations, this module would provide a dedicated place for their definitions.
5.  **`src/data.py`**: This module is a placeholder for more sophisticated data handling. In the current setup, environments are created directly using `gymnasium`. For environments requiring custom wrappers, specialized data loading, or replay buffers, this module would be expanded.
6.  **`src/utils.py`**: A collection of utility functions that support the agents and experiments. This includes the `train_*_agent` functions (e.g., `train_reinforce_agent`, `train_ppo_agent`), functions for comparative analysis (`compare_policy_gradient_variants`), hyperparameter sensitivity analysis (`hyperparameter_sensitivity_analysis`), curriculum learning demonstrations (`curriculum_learning_demo`), and plotting routines (`plot_policy_gradient_comparison`).

## Theoretical Foundations

### 5.1 The Policy Gradient Theorem

The Policy Gradient Theorem is the cornerstone of policy-based reinforcement learning. It provides a way to calculate the gradient of the expected return $J(\theta)$ with respect to the policy parameters $\theta$ without explicit knowledge of the environment dynamics. The objective is to find $\theta$ that maximizes $J(\theta)$.

The fundamental form of the theorem states:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) \right]$$

where:
*   $\rho^\pi$ is the stationary distribution of states under policy $\pi$.
*   $\pi_\theta(a|s)$ is the probability of taking action $a$ in state $s$ under policy $\theta$.
*   $Q^\pi(s,a)$ is the action-value function, representing the expected return from taking action $a$ in state $s$ and thereafter following policy $\pi$.

In practice, $Q^\pi(s,a)$ is often replaced by $G_t$ (the total return from time step $t$) in Monte Carlo methods, or by a learned value function estimate in actor-critic methods.

### 5.2 REINFORCE Algorithm

REINFORCE, or Monte Carlo Policy Gradient, is a direct application of the Policy Gradient Theorem. For each episode, the agent collects a trajectory of states, actions, and rewards. At the end of the episode, the return $G_t$ is calculated for each time step $t$, and the policy parameters are updated via stochastic gradient ascent:

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

where $\alpha$ is the learning rate. While conceptually simple, REINFORCE suffers from high variance in its gradient estimates due to the noisy Monte Carlo returns, leading to slow convergence.

### 5.3 Variance Reduction: Baselines

To mitigate the high variance of REINFORCE, a common technique is to subtract a baseline function $b(s)$ from the return $G_t$. This operation does not introduce bias into the gradient estimate, provided the baseline does not depend on the action $a_t$. The state-value function $V^\pi(s_t)$ is a natural choice for a baseline:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) (G_t - V^\pi(s_t)) \right]$$

The term $(G_t - V^\pi(s_t))$ is an estimate of the advantage function $A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$. This advantage indicates how much better an action $a_t$ is compared to the average action in state $s_t$. By subtracting the baseline, the magnitude of the gradient updates is reduced for actions that are only slightly better or worse than average, thus lowering variance.

### 5.4 Actor-Critic Methods

Actor-Critic methods combine the strengths of policy-based and value-based approaches. They maintain two distinct components:

*   **Actor (Policy)**: A parameterized policy $\pi_\theta(a|s)$ that selects actions.
*   **Critic (Value Function)**: A parameterized value function $V_\phi(s)$ (or $Q_\phi(s,a)$) that estimates the value of states or state-action pairs.

The critic learns to estimate the value function, and this estimate is used to construct a low-variance estimate of the advantage function for updating the actor. The actor is updated using the critic's feedback, often through the TD error as an advantage estimate:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) (r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)) \right]$$

Here, the term $(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))$ serves as the TD error and acts as the advantage estimate, providing a more immediate and lower-variance signal than Monte Carlo returns.

### 5.5 Proximal Policy Optimization (PPO)

PPO is one of the most popular and robust policy gradient algorithms, designed to achieve a balance between sample efficiency and stability. It addresses the challenge of step size in policy gradient methods by introducing a clipped surrogate objective function. This objective prevents new policies from straying too far from the old policy, which can otherwise lead to destructive updates.

The PPO clipped objective is given by:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

where:
*   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policies.
*   $\hat{A}_t$ is the advantage estimate at time $t$, often calculated using Generalized Advantage Estimation (GAE).
*   $\epsilon$ is a small hyperparameter (e.g., 0.1 or 0.2) that defines the clipping range.

The $\min$ operator ensures that the policy updates are conservative. If the probability ratio $r_t(\theta)$ becomes too large (i.e., the new policy assigns much higher probability to an action than the old policy), it is clipped, effectively preventing excessively large updates that could destabilize training. PPO also often includes an entropy bonus in its objective to encourage exploration.

## Usage

### Running Experiments via `main.py`

The `main.py` script serves as the primary entry point for orchestrating all experiments. It allows running various algorithms and analyses with configurable parameters.

```bash
# Run all basic algorithms and advanced analyses with default episodes (from src/config.py)
python main.py

# Run in quick mode (fewer episodes for faster demonstrations)
python main.py --quick

# Run only basic algorithms (REINFORCE, Actor-Critic, PPO)
python main.py --algorithms-only

# Run only advanced analyses (comparisons, hyperparameter sensitivity, curriculum learning)
python main.py --analyses-only
```

### Interacting with `notebooks/main.ipynb`

The `notebooks/main.ipynb` file is the central hub for interactive experimentation, visualization, and detailed output. It imports all necessary components from the `src/` directory, allowing you to run and modify experiments directly within the notebook environment. This is the recommended way to explore the code and results.

```bash
jupyter notebook notebooks/main.ipynb
```

## Key Features

### Implemented Algorithms

*   **REINFORCE**: Classic Monte Carlo policy gradient.
*   **REINFORCE with Baseline**: Variance reduction through a learned value function.
*   **Actor-Critic**: TD-based policy and value learning for improved stability.
*   **PPO (Proximal Policy Optimization)**: State-of-the-art algorithm balancing sample efficiency and robust updates.
*   **Continuous Control**: Support for continuous action spaces using Gaussian policies (demonstrated with PPO).

### Advanced Analysis & Tools

*   **Comparative Analysis**: Direct performance comparison of different policy gradient variants.
*   **Hyperparameter Sensitivity**: Experiments to understand the impact of learning rates, discount factors, and other hyperparameters.
*   **Curriculum Learning**: Demonstration of a basic curriculum setup for progressive learning.
*   **Modular Architecture**: Clean separation of concerns (config, models, agents, utilities) for better code organization.
*   **Reproducibility**: Global random seed management for consistent experiment outcomes.

## Dependencies

The project requires the following Python packages. These are listed in `requirements.txt`.

*   `torch >= 2.0`
*   `gymnasium >= 0.29.1`
*   `numpy >= 1.26.0`
*   `matplotlib >= 3.8.0`
*   `seaborn >= 0.13.0`
*   `pandas >= 2.1.0`

## Environment Compatibility

All implementations are tested and compatible with standard Gymnasium environments, including:

*   **Discrete Action Spaces**: `CartPole-v1` (default for many discrete control examples).
*   **Continuous Action Spaces**: `Pendulum-v1` (default for continuous control examples).

## Performance Benchmarks

Typical performance on `CartPole-v1` (achieving an average reward of 195+ over 100 consecutive episodes is considered "solved"):

*   **REINFORCE**: Generally reaches average rewards in the range of ~100-150.
*   **REINFORCE with Baseline**: Improves stability, often reaching ~150-180 average rewards.
*   **Actor-Critic**: Further enhanced stability and performance, typically reaching ~180-200 average rewards.
*   **PPO**: Consistently solves `CartPole-v1`, achieving 195+ average reward reliably.

## Extension Points

The modular design facilitates easy extension and experimentation:

1.  **New Algorithms**: Implement new policy gradient or other RL agent classes within `src/agents.py`.
2.  **New Network Architectures**: Define custom neural network models in `src/model.py`.
3.  **Custom Environments**: Integrate custom Gymnasium environments or wrappers by modifying `src/data.py` (if needed) and adjusting training calls in `src/utils.py`.
4.  **Advanced Analyses**: Add new evaluation metrics, visualization types, or experiment setups in `src/utils.py`.
5.  **Hyperparameter Tuning**: Easily adjust hyperparameters in `src/config.py`.

## Testing

To ensure the correctness of the modular components, you can run various checks:

```bash
# Static type checking (if mypy is installed)
mypy src/

# Run the main script to execute all demonstrations and generate plots
python main.py

# Open the notebook for interactive testing and visualization
jupyter notebook notebooks/main.ipynb
```

## Results and Visualizations

After running `main.py` or the `notebooks/main.ipynb`, you will find generated outputs in the following directories:

*   **`visualizations/`**: Contains various plots, including learning curves, performance comparisons, and hyperparameter sensitivity plots. These are generated by `src/utils.py` and saved for easy review.
*   **`results/`**: (Placeholder) Can be used to store detailed training logs, raw performance data, and experiment summaries.
*   **`logs/`**: (Placeholder) For system-level or verbose debugging logs if configured.

## License

This project is part of the Deep Reinforcement Learning course materials and is provided under the MIT License.
