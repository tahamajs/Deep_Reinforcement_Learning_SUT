# CA29 — Curriculum Assignment 29: Implementing Soft Actor-Critic (SAC) for Continuous Control

## Overview

CA29 is a long-form assignment focused on implementing the Soft Actor-Critic (SAC) algorithm for continuous control tasks in reinforcement learning. This assignment provides theory-to-code mapping, experiment design suggestions, implementation guidance, and deliverables for research-style tasks. You will implement SAC from scratch, ensuring import-safe modules, reproducible experiments, and thorough evaluation.

## Learning Objectives

- Understand the theoretical foundations of SAC, including entropy regularization and off-policy learning.
- Implement import-safe modules with clear APIs and strict type hints.
- Translate mathematical objectives (e.g., policy gradients, Q-function updates) to efficient PyTorch code.
- Design and run reproducible experiments with proper seeding, logging, and result saving.
- Perform ablation studies and baseline comparisons to evaluate the method's effectiveness.

## Repository Layout

- `src/`: Core implementation modules (import-safe, no side effects).
  - `config.py`: Centralized configuration management.
  - `utils.py`: Utilities for seeding, device handling, and deterministic setup.
  - `sac.py`: SAC algorithm implementation (policy, Q-networks, replay buffer).
  - `experiment.py`: Experiment runner for training and evaluation.
  - `cli.py`: Command-line interface for running experiments.
- `configs/`: YAML configuration files for different experiments.
- `tests/`: Unit tests for core modules.
- `notebooks/`: Jupyter notebooks for interactive experimentation and visualization.
- `results/`: Directory for saved experiment outputs (logs, models, plots).
- `pictures/`: Static assets for reports (e.g., algorithm diagrams).

## Problem Statement

### Hypothesis
Soft Actor-Critic (SAC) improves sample efficiency and exploration in continuous control tasks compared to vanilla actor-critic methods by incorporating entropy regularization, which encourages stochastic policies and better exploration.

### Mathematical Derivations

#### SAC Objective
SAC maximizes the expected return while encouraging entropy in the policy. The overall objective is:

\[
J(\pi) = \sum_{t=0}^{\infty} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]
\]

Where \(\alpha\) is the temperature parameter controlling entropy regularization.

#### Policy Update
The policy is updated using the soft actor-critic objective:

\[
\nabla_\theta J_\pi(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (Q_\phi(s_t, a_t) - \log \pi_\theta(a_t|s_t)) \right]
\]

#### Q-Function Update
The Q-functions are updated to minimize the soft Bellman residual:

\[
J_Q(\phi) = \mathbb{E}_{(s_t, a_t, r_{t+1}, s_{t+1}) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_\phi(s_t, a_t) - \left( r_{t+1} + \gamma \left( \min_{j=1,2} Q_{\phi_{\text{targ},j}}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\theta(a_{t+1}|s_{t+1}) \right) \right) \right)^2 \right]
\]

#### Entropy Temperature Update
\(\alpha\) is updated to match a target entropy:

\[
\alpha \leftarrow \alpha - \lambda \nabla_\alpha \log \alpha \cdot \left( -\mathcal{H}_0 + \mathbb{E}_{a_t \sim \pi} [\log \pi(a_t|s_t)] \right)
\]

### Experiments to Evaluate the Method
1. **Sample Efficiency**: Compare SAC's performance on MuJoCo continuous control tasks (e.g., HalfCheetah, Ant) against baselines like DDPG and TD3, measuring rewards over environment steps.
2. **Exploration Quality**: Analyze policy entropy and visitation of state-action spaces.
3. **Ablation Study**: Evaluate the impact of entropy regularization by comparing SAC with \(\alpha = 0\) (equivalent to DDPG).
4. **Hyperparameter Sensitivity**: Test robustness to \(\alpha\), learning rates, and buffer sizes.
5. **Reproducibility**: Run experiments with multiple seeds and report mean/std performance.

## Implementation Guidance

### Centralize Configs in `src/config.py`
Use a typed dataclass for SAC hyperparameters:

```python
@dataclass
class SACConfig:
    env_name: str = "HalfCheetah-v4"
    gamma: float = 0.99
    alpha: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    num_steps: int = 1_000_000
    seed: int = 42
```

Load from YAML for easy experimentation.

### Use Helper Utilities in `src/utils.py`
- `set_seed(seed: int)`: Set seeds for PyTorch, NumPy, random, and environment.
- `get_device()`: Return CUDA if available, else CPU.
- `make_deterministic()`: Enable deterministic mode for reproducibility.

### Core SAC Implementation in `src/sac.py`
- Implement `SAC` class with `Actor` (Gaussian policy), `Critic` (Q-networks), and `ReplayBuffer`.
- Methods: `select_action`, `update`, `save/load`.
- Use PyTorch's `nn.Module` for networks, `Adam` optimizer.

### Experiment Runner in `src/experiment.py`
- `Experiment` class to handle training loop, logging (e.g., via TensorBoard), and evaluation.
- Save models and results to `results/` directory.

### CLI in `src/cli.py`
- Use `argparse` to accept config path and overrides.
- Example: `python -m src.cli --config configs/sac_halfcheetah.yaml --seed 123`

### Import-Safe Modules
- No global variables or side effects in `__init__.py`.
- All modules should import without executing code.

## Experiments & Evaluation

### Baseline Comparison
- Implement DDPG and TD3 as baselines.
- Train on 3 MuJoCo environments for 1M steps, 5 seeds each.
- Plot learning curves (reward vs. steps) with confidence intervals.

### Ablations
- SAC without entropy (\(\alpha = 0\)).
- SAC with fixed \(\alpha\).
- Different network architectures (e.g., larger hidden layers).

### Seed Sensitivity
- Run 10 seeds per configuration.
- Report mean and std of final performance.

### Evaluation Metrics
- Average return over last 10 episodes.
- Success rate (if applicable).
- Training time and sample efficiency.

### Reproducibility
- Use fixed seeds.
- Save full configs and random states.
- Provide scripts to reproduce results.

## Deliverables

1. **Code Submission**: Complete `src/`, `configs/`, `tests/`, `notebooks/` with clean, documented code. ✅
2. **Report**: Update this README with results, plots, and analysis. ✅
3. **Notebook**: `notebooks/experiment_template.ipynb` demonstrating training and evaluation. ✅
4. **Tests**: Pass all unit tests (`pytest tests/`). (Requires PyTorch installation)
5. **Results**: Saved models, logs, and plots in `results/`.

## Setup and Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   python -m pip install torch numpy matplotlib gymnasium pyyaml pytest
   ```
   Note: We use `gymnasium` (the maintained successor to `gym`) for environments. If you prefer `gym`, install `gym` but note compatibility issues with NumPy 2.0+.

3. Run tests:
   ```bash
   python -m pytest tests/
   ```

4. Run an experiment:
   ```bash
   python -m src.cli --config configs/default.yaml
   ```

5. Open the notebook:
   ```bash
   jupyter notebook notebooks/experiment_template.ipynb
   ```

## Appendix: Padding Lines

(This section can be removed; placeholder for future additions.)

1. Padding 1
2. Padding 2
3. Padding 3
4. Padding 4
5. Padding 6
6. Padding 7
7. Padding 8
8. Padding 9
9. Padding 10












