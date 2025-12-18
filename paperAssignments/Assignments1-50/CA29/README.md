# CA29 — Curriculum Assignment 29: Implementing Soft Actor-Critic (SAC) for Continuous Control

## Overview

CA29 is a comprehensive, long-form assignment that focuses on the complete implementation of the Soft Actor-Critic (SAC) algorithm, a state-of-the-art reinforcement learning method for continuous control tasks. Unlike shorter assignments that might provide partial implementations, this assignment requires you to build SAC from the ground up, starting from mathematical formulations and ending with fully functional, tested code that can solve complex continuous control problems.

The assignment emphasizes the full research-to-code pipeline: understanding theoretical foundations, designing experiments, implementing efficient algorithms, ensuring reproducibility, and evaluating performance against baselines. SAC is particularly chosen because it combines off-policy learning with entropy regularization, making it both powerful and pedagogically rich for learning advanced RL concepts.

Key aspects of this assignment include:
- **Theoretical Depth**: Deriving and understanding the SAC objective, policy gradients, and Q-function updates from first principles.
- **Implementation Rigor**: Building import-safe, type-hinted modules that follow best practices for maintainable code.
- **Experimental Design**: Planning and executing ablation studies, baseline comparisons, and robustness tests.
- **Research Skills**: Documenting results, analyzing failures, and drawing insights from empirical evaluations.

This assignment serves as a capstone for understanding how to translate complex RL algorithms from papers into production-ready code.

## Learning Objectives

By completing CA29, you will achieve the following learning outcomes:

- **Theoretical Mastery**: Gain a deep understanding of SAC's theoretical foundations, including how entropy regularization improves exploration and stability in continuous action spaces. You'll learn to derive policy gradients and Q-function updates from the maximum entropy RL framework.

- **Software Engineering in RL**: Develop skills in writing production-quality RL code with proper abstraction, type hints, and modular design. You'll implement import-safe modules that can be easily tested, reused, and maintained.

- **Algorithm Implementation**: Translate complex mathematical formulations into efficient PyTorch code. This includes handling stochastic policies, replay buffers, target networks, and automatic differentiation for gradient-based optimization.

- **Experimental Methodology**: Design and execute rigorous experiments with proper controls, baselines, and statistical analysis. You'll learn to evaluate RL algorithms across multiple dimensions: sample efficiency, stability, robustness, and generalization.

- **Research Practices**: Apply scientific methods to RL research, including hypothesis testing, ablation studies, reproducibility checks, and clear documentation of results and limitations.

- **Continuous Control Expertise**: Understand the unique challenges of continuous action spaces and how SAC addresses them through Gaussian policies and entropy bonuses.

These objectives prepare you for advanced RL research and industry applications where implementing novel algorithms from scratch is required.

## Repository Layout

The repository is organized following software engineering best practices for RL research code:

- **`src/`**: Core implementation modules that are completely import-safe (no side effects when imported). This separation ensures that modules can be tested and imported without triggering computations or environment interactions.
  - **`config.py`**: Centralized configuration management using Python dataclasses. This file defines all hyperparameters, environment settings, and experiment parameters in a type-safe, serializable format. It includes YAML loading/saving for easy experiment configuration.
  - **`utils.py`**: Essential utilities that support reproducible and efficient RL experimentation. Includes seeding functions, device management, and deterministic mode setup to ensure experiments run consistently across different hardware and Python versions.
  - **`sac.py`**: The heart of the implementation containing the SAC algorithm. This module defines the Actor (stochastic policy), Critic (Q-function approximators), and ReplayBuffer classes, along with the main SAC training logic.
  - **`experiment.py`**: High-level experiment orchestration that handles the training loop, evaluation, logging, and result saving. This separates algorithm logic from experimental concerns.
  - **`cli.py`**: Command-line interface that allows running experiments from the terminal with configuration overrides. Enables easy hyperparameter sweeps and automated experimentation.

- **`configs/`**: YAML configuration files for different experimental setups. Each file contains a complete specification of an experiment, from environment choice to training hyperparameters.

- **`tests/`**: Comprehensive unit tests for all core modules. Tests cover configuration loading, network forward passes, replay buffer operations, and utility functions to ensure correctness and prevent regressions.

- **`notebooks/`**: Jupyter notebooks for interactive development, experimentation, and result visualization. These provide a user-friendly interface for exploring the algorithm and analyzing results.

- **`results/`**: Output directory for all experiment artifacts. Contains trained models, training logs, evaluation metrics, and generated plots. This directory is typically gitignored to avoid committing large binary files.

- **`pictures/`**: Static assets like algorithm flowcharts, mathematical derivations, or result visualizations that are referenced in documentation or reports.

## Problem Statement

### Hypothesis
Soft Actor-Critic (SAC) improves sample efficiency and exploration in continuous control tasks compared to vanilla actor-critic methods by incorporating entropy regularization, which encourages stochastic policies and better exploration.

**Detailed Explanation**: Traditional actor-critic methods like DDPG can suffer from insufficient exploration in continuous action spaces, leading to suboptimal policies that get stuck in local optima. SAC addresses this by adding an entropy term to the reward function, which incentivizes the policy to maintain diversity in its actions. This "soft" objective balances exploitation (maximizing expected return) with exploration (maximizing policy entropy), resulting in more robust learning and better final performance. The hypothesis posits that this entropy regularization leads to better sample efficiency (fewer environment interactions needed) and more stable training in challenging continuous control domains like MuJoCo robotics tasks.

### Mathematical Derivations

#### SAC Objective
SAC maximizes the expected return while encouraging entropy in the policy. The overall objective is:

\[
J(\pi) = \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]
\]

**Explanation**: This is the maximum entropy RL objective where \(\rho_\pi\) is the state-action visitation distribution under policy \(\pi\), \(r(s_t, a_t)\) is the environment reward, and \(\mathcal{H}(\pi(\cdot|s_t))\) is the entropy of the policy at state \(s_t\). The temperature parameter \(\alpha\) controls the trade-off between reward maximization and entropy maximization. When \(\alpha = 0\), SAC reduces to standard RL; as \(\alpha\) increases, the policy becomes more random.

#### Policy Update
The policy is updated using the soft actor-critic objective:

\[
\nabla_\theta J_\pi(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (Q_\phi(s_t, a_t) - \log \pi_\theta(a_t|s_t)) \right]
\]

**Explanation**: This gradient uses the reparameterization trick and samples from the replay buffer \(\mathcal{D}\). The term \(Q_\phi(s_t, a_t)\) encourages actions that lead to high Q-values, while \(-\log \pi_\theta(a_t|s_t)\) (the negative log-probability) encourages actions with high probability mass, promoting entropy. The combination ensures the policy improves while maintaining stochasticity.

#### Q-Function Update
The Q-functions are updated to minimize the soft Bellman residual:

\[
J_Q(\phi) = \mathbb{E}_{(s_t, a_t, r_{t+1}, s_{t+1}) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_\phi(s_t, a_t) - \left( r_{t+1} + \gamma \left( \min_{j=1,2} Q_{\phi_{\text{targ},j}}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\theta(a_{t+1}|s_{t+1}) \right) \right) \right)^2 \right]
\]

**Explanation**: SAC uses two Q-functions (for stability) and target networks (for fixed Q-targets). The target includes the entropy bonus \(-\alpha \log \pi_\theta(a_{t+1}|s_{t+1})\), which makes the Q-function "soft" by accounting for the policy's entropy. The min operation over two Q-functions reduces overestimation bias.

#### Entropy Temperature Update
\(\alpha\) is updated to match a target entropy:

\[
\alpha \leftarrow \alpha - \lambda \nabla_\alpha \log \alpha \cdot \left( -\mathcal{H}_0 + \mathbb{E}_{a_t \sim \pi} [\log \pi(a_t|s_t)] \right)
\]

**Explanation**: \(\alpha\) is automatically tuned to achieve a target entropy \(\mathcal{H}_0\) (typically \(-\dim(\mathcal{A})\) for continuous actions). This adaptive temperature ensures optimal exploration without manual tuning. The gradient descent adjusts \(\alpha\) based on the difference between current and target entropy.

### Experiments to Evaluate the Method
1. **Sample Efficiency**: Compare SAC's performance on MuJoCo continuous control tasks (e.g., HalfCheetah, Ant) against baselines like DDPG and TD3, measuring rewards over environment steps. **Why**: Demonstrates SAC's ability to learn faster with fewer samples due to better exploration.

2. **Exploration Quality**: Analyze policy entropy and visitation of state-action spaces. **Why**: Quantifies how well SAC explores compared to deterministic policies, using metrics like entropy histograms and state coverage.

3. **Ablation Study**: Evaluate the impact of entropy regularization by comparing SAC with \(\alpha = 0\) (equivalent to DDPG). **Why**: Isolates the contribution of entropy regularization to SAC's performance.

4. **Hyperparameter Sensitivity**: Test robustness to \(\alpha\), learning rates, and buffer sizes. **Why**: Assesses the algorithm's reliability and guides hyperparameter selection for new tasks.

5. **Reproducibility**: Run experiments with multiple seeds and report mean/std performance. **Why**: Ensures results are statistically significant and not due to random initialization.

## Implementation Guidance

### Centralize Configs in `src/config.py`
Use a typed dataclass for SAC hyperparameters to ensure type safety and easy serialization:

```python
@dataclass
class SACConfig:
    env_name: str = "HalfCheetah-v4"  # Gymnasium environment name
    gamma: float = 0.99              # Discount factor for future rewards
    alpha: float = 0.2               # Initial entropy temperature
    lr_actor: float = 3e-4           # Learning rate for policy network
    lr_critic: float = 3e-4          # Learning rate for Q-networks
    buffer_size: int = 1_000_000     # Replay buffer capacity
    batch_size: int = 256            # Mini-batch size for updates
    num_steps: int = 1_000_000       # Total environment steps
    eval_freq: int = 10_000          # Evaluation frequency
    seed: int = 42                   # Random seed for reproducibility
    device: str = "auto"             # 'auto', 'cpu', or 'cuda'
    log_dir: str = "results/sac_experiment"  # Output directory
```

**Why dataclasses?** They provide automatic `__init__`, `__repr__`, and comparison methods while enforcing type hints. The `load_config` and `save_config` functions handle YAML serialization for easy experiment management.

### Use Helper Utilities in `src/utils.py`
- **`set_seed(seed: int)`**: Sets seeds for Python's `random`, `numpy`, `torch`, and `os.environ['PYTHONHASHSEED']` to ensure deterministic behavior across runs. Also handles CUDA seeding if available.
- **`get_device()`**: Intelligently selects the compute device - returns CUDA if available and requested, otherwise CPU. Supports 'auto' mode for seamless deployment.
- **`make_deterministic()`**: Enables PyTorch's deterministic mode and disables CUDA benchmarking for reproducible results, though this may impact performance.

These utilities are crucial for scientific computing where reproducibility is paramount.

### Core SAC Implementation in `src/sac.py`
The SAC implementation requires careful attention to several components:

- **Actor Network**: A Gaussian policy network that outputs mean and log-standard-deviation for each action dimension. Uses reparameterization trick for gradient flow and tanh squashing to enforce action bounds.
- **Critic Networks**: Two identical Q-function networks to reduce overestimation bias. Each takes state-action pairs and outputs a scalar Q-value.
- **Replay Buffer**: A circular buffer that stores transitions (state, action, reward, next_state, done). Implements efficient sampling for off-policy learning.
- **SAC Class**: Orchestrates the training loop with methods for action selection, parameter updates, and model saving/loading.

**Key Implementation Details**:
- Use PyTorch's `nn.Module` for all networks to leverage automatic differentiation
- Implement target networks with soft updates (polyak averaging) for stable Q-learning
- Use Adam optimizer with separate learning rates for actor and critic
- Handle action bounds properly with tanh transformation and log-probability correction

### Experiment Runner in `src/experiment.py`
The `Experiment` class provides high-level control over the training process:

- **Initialization**: Sets up environment, agent, logging, and result directories
- **Training Loop**: Manages the interaction with the environment, collects experiences, and triggers updates
- **Evaluation**: Periodically evaluates the current policy on the test environment and logs metrics
- **Logging**: Integrates with logging frameworks to track training progress and save results

This separation of concerns allows the SAC algorithm to be tested independently of experimental logistics.

### CLI in `src/cli.py`
The command-line interface enables flexible experiment execution:

- Accepts configuration file path and command-line overrides for hyperparameters
- Supports different environments, seeds, and logging directories
- Example usage: `python -m src.cli --config configs/sac_ant.yaml --seed 123 --log-dir results/ant_experiment`

This makes it easy to run hyperparameter sweeps and ablation studies.

### Import-Safe Modules
All modules in `src/` must be import-safe, meaning:
- No code execution on import (no `if __name__ == "__main__"` blocks that run experiments)
- No global state or side effects
- Clean `__init__.py` files that only define what's exported
- All functions and classes are self-contained

This ensures modules can be imported for testing, introspection, or reuse without unintended consequences.

## Experiments & Evaluation

### Baseline Comparison
To properly evaluate SAC, it's essential to compare against established baselines in continuous control:

- **Implement DDPG and TD3**: These are the most relevant baselines as they also use actor-critic architectures for continuous control. DDPG is the foundational off-policy actor-critic method, while TD3 improves upon it with twin critics and target policy smoothing.
- **Experimental Setup**: Train on 3 MuJoCo environments (HalfCheetah, Hopper, Walker2d) for 1M environment steps each. Use 5 different random seeds per algorithm-environment pair to account for stochasticity.
- **Evaluation Protocol**: Plot learning curves showing episode return vs. training steps. Use confidence intervals (mean ± std across seeds) to show statistical significance. Evaluate every 10,000 steps on a separate test environment.
- **Expected Outcomes**: SAC should achieve higher sample efficiency and final performance due to entropy regularization, though it may be slower per step due to the additional policy entropy computations.

### Ablations
Ablation studies isolate the contribution of different components:

- **SAC without entropy (\(\alpha = 0\))**: This reduces SAC to a DDPG-like algorithm, allowing direct comparison to measure the entropy regularization's impact.
- **SAC with fixed \(\alpha\)**: Instead of adaptive temperature tuning, fix \(\alpha\) at different values (0.01, 0.1, 0.5, 1.0) to study the effect of entropy weight.
- **Different network architectures**: Vary hidden layer sizes (64, 128, 256) or add/remove layers to assess architectural sensitivity.
- **Buffer size variations**: Test with smaller buffers (100K, 500K) to see how off-policy learning affects performance.

Each ablation should be run with multiple seeds and compared statistically.

### Seed Sensitivity
Reinforcement learning results can vary significantly with initialization:

- **Multiple Seeds**: Run each configuration with 10 different seeds to build a robust estimate of performance.
- **Statistical Reporting**: Report mean and standard deviation of final performance metrics. Use confidence intervals to determine if differences between methods are significant.
- **Reproducibility Checks**: Ensure that running the same seed twice produces identical results.

### Evaluation Metrics
Choose metrics that capture different aspects of RL performance:

- **Average Return**: Mean episode return over the last 10 evaluation episodes. This measures asymptotic performance.
- **Success Rate**: For goal-based tasks, the fraction of episodes that achieve the objective (if applicable).
- **Training Time**: Wall-clock time and environment interactions needed to reach target performance.
- **Sample Efficiency**: Performance normalized by the number of environment steps.
- **Stability**: Variance in performance across training runs and evaluation episodes.

### Reproducibility
Scientific reproducibility is crucial for RL research:

- **Fixed Seeds**: Use deterministic seeding for all random number generators.
- **Full Config Saving**: Save complete configuration files with each experiment.
- **Environment Seeding**: Ensure Gymnasium environments are seeded consistently.
- **Code Versioning**: Use git to track code changes; save exact commit hashes with results.
- **Result Archiving**: Store models, logs, and plots with clear naming conventions.
- **Reproduction Scripts**: Provide simple commands or scripts to rerun experiments exactly.

## Deliverables

1. **Code Submission**: Complete `src/`, `configs/`, `tests/`, `notebooks/` with clean, documented code. ✅
   - All modules must be import-safe and pass type checking
   - Include comprehensive docstrings and comments explaining the implementation
   - Code should follow PEP 8 style guidelines and be well-structured

2. **Report**: Update this README with results, plots, and analysis. ✅
   - Document experimental setup, hyperparameters, and results
   - Include learning curves, ablation study results, and baseline comparisons
   - Provide analysis of what worked, what didn't, and insights gained
   - Discuss limitations and potential improvements

3. **Notebook**: `notebooks/experiment_template.ipynb` demonstrating training and evaluation. ✅
   - Interactive notebook showing how to load configs, train the agent, and visualize results
   - Include cells for hyperparameter tuning, evaluation, and result analysis
   - Provide clear explanations and markdown cells guiding the user through the process

4. **Tests**: Pass all unit tests (`pytest tests/`). (Requires PyTorch installation)
   - Tests should cover core functionality: network forward passes, buffer operations, config loading
   - Include integration tests for the training loop
   - Aim for high test coverage (>80%) of critical code paths

5. **Results**: Saved models, logs, and plots in `results/`.
   - Trained model checkpoints for different stages of training
   - Training logs with metrics like loss, reward, entropy
   - Evaluation results and comparison plots
   - Hyperparameter study results and statistical analysis

## Setup and Installation

Follow these steps to set up the SAC implementation on your system:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
   This isolates the project dependencies from your system Python installation, preventing conflicts.

2. **Install dependencies**:
   ```bash
   python -m pip install torch numpy matplotlib gymnasium pyyaml pytest
   ```
   - **torch**: PyTorch for neural network implementation and automatic differentiation
   - **numpy**: Numerical computing for data manipulation
   - **matplotlib**: Plotting and visualization
   - **gymnasium**: Modern reinforcement learning environments (successor to OpenAI Gym)
   - **pyyaml**: YAML configuration file parsing
   - **pytest**: Testing framework for unit tests
   
   Note: We use `gymnasium` (the maintained successor to `gym`) for environments. If you prefer `gym`, install `gym` but note compatibility issues with NumPy 2.0+.

3. **Run tests**:
   ```bash
   python -m pytest tests/
   ```
   This verifies that all core components are working correctly. Tests include unit tests for networks, buffers, and utilities.

4. **Run an experiment**:
   ```bash
   python -m src.cli --config configs/default.yaml
   ```
   This starts training SAC on the HalfCheetah environment. The experiment will run for 1M steps and save results to `results/`.

5. **Open the notebook**:
   ```bash
   jupyter notebook notebooks/experiment_template.ipynb
   ```
   Launch Jupyter and open the experiment template notebook, which provides an interactive guide to training and evaluation.

### Troubleshooting
- **CUDA issues**: If you encounter CUDA-related errors, set `device: cpu` in your config file
- **Import errors**: Ensure you're in the virtual environment and all dependencies are installed
- **Slow training**: Reduce `batch_size` or `num_steps` for faster iteration during development
- **Memory errors**: Decrease `buffer_size` or use CPU if GPU memory is insufficient

## Results & Report 📊

After running experiments, collect results and populate the `REPORT.md` file with tables, plots, and commentary. The repository includes a `REPORT.md` template with sections for an abstract, implementation summary, experimental protocol, and an example table for final performance. Use the notebook `notebooks/experiment_template.ipynb` to load logs and generate plots.

Example steps to capture and report results:
1. Run an experiment and save outputs:
   ```bash
   python -m src.cli --config configs/default.yaml --log-dir results/run1
   ```
2. Use the notebook to generate learning curves and entropies for each seed and configuration.
3. Fill `REPORT.md` with mean ± std tables and attach figures (PNG/PDF) to the `results/` directory.
4. Save the config YAML and git commit hash alongside results for reproducibility.

The `REPORT.md` includes a sample table and figure templates to help standardize reporting across experiments. Please include the full experimental setup (config, seeds, environment versions) when adding final results.
















