# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 2: Policy Gradient Methods

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment explores **policy gradient methods**, a class of reinforcement learning algorithms that directly optimize the policy's parameters by performing gradient ascent on the expected return. Unlike value-based methods (like Q-learning) that learn a value function and derive a policy from it, policy gradient methods learn the policy directly. This makes them particularly effective for continuous action spaces and stochastic policies.

You will implement the foundational REINFORCE algorithm and investigate powerful variance reduction techniques that are crucial for making policy gradients work in practice, including reward-to-go, baselines, and Generalized Advantage Estimation (GAE).

## Learning Objectives

- Understand the theoretical foundation of the **Policy Gradient Theorem**.
- Implement the **REINFORCE** algorithm (Monte Carlo Policy Gradient).
- Implement and analyze key **variance reduction techniques**:
    - Reward-to-Go
    - Neural Network Baselines
    - Generalized Advantage Estimation (GAE)
- Understand the trade-offs between bias and variance in policy gradient estimators.
- Gain practical experience with policy gradients on continuous control tasks.

## Key Concepts Explained

### 1. Policy Gradient Theorem
The core idea is to adjust the policy's parameters `θ` in the direction that increases the expected total reward `J(θ)`. The Policy Gradient Theorem provides a way to compute this gradient:
```
∇J(θ) = E_τ [ (Σ_t ∇_θ log π_θ(a_t|s_t)) * R(τ) ]
```
- `π_θ(a_t|s_t)`: The policy (probability of taking action `a_t` in state `s_t`).
- `R(τ)`: The total reward for a trajectory `τ`.
- `∇_θ log π_θ(a_t|s_t)`: The "score function." It tells us how to change `θ` to increase the probability of action `a_t`.

This means we can increase the probability of actions that lead to high rewards and decrease the probability of actions that lead to low rewards.

### 2. REINFORCE (Monte Carlo Policy Gradient)
REINFORCE is the most basic policy gradient algorithm. It uses Monte Carlo estimation (i.e., full trajectory rollouts) to compute the gradient:
1.  Collect a batch of trajectories by running the current policy `π_θ`.
2.  For each trajectory, compute the total return `R(τ)`.
3.  Estimate the gradient using the sample average: `∇J(θ) ≈ (1/N) Σ_i [ (Σ_t ∇_θ log π_θ(a_it|s_it)) * R(τ_i) ]`.
4.  Update the policy parameters: `θ ← θ + α∇J(θ)`.

**Limitation**: The gradient estimate is very noisy (high variance) because the total return `R(τ)` is influenced by many actions, making it a poor credit assignment mechanism.

### 3. Variance Reduction: Reward-to-Go
**Problem**: An action at timestep `t` can only affect rewards from `t` onwards. Using the full return `R(τ)` to update the policy for an action at `t` introduces noise from past rewards.
**Solution**: Replace the total return `R(τ)` with the **reward-to-go**, which is the sum of rewards from timestep `t` to the end of the episode.
```
Q(s_t, a_t) = Σ_{t'=t to T} r(s_t', a_t')
```
This significantly reduces variance by improving credit assignment.

### 4. Variance Reduction: Baselines
**Idea**: We can subtract a baseline `b(s_t)` from the return term without changing the expected value of the gradient (it remains unbiased).
```
∇J(θ) ∝ E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) * (Q(s_t, a_t) - b(s_t)) ]
```
A good baseline is the **value function `V(s_t)`**, which is the expected return from state `s_t`. The term `A(s_t, a_t) = Q(s_t, a_t) - V(s_t)` is called the **Advantage Function**. It measures whether an action was better or worse than average for that state.
- If `A > 0`, increase the probability of that action.
- If `A < 0`, decrease the probability of that action.
In this homework, you will train a separate neural network to approximate `V(s_t)` to serve as the baseline.

### 5. Generalized Advantage Estimation (GAE)
GAE provides a sophisticated way to estimate the advantage function that balances bias and variance. It uses a parameter `λ` (lambda) to interpolate between a simple one-step TD-error (low variance, high bias) and the Monte Carlo estimate (high variance, low bias).
```
A_GAE(s_t, a_t) = Σ_{l=0 to ∞} (γλ)^l * δ_{t+l}
where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD-error.
```
- `λ = 0`: Results in the simple TD-advantage estimate `A = r_t + γV(s_{t+1}) - V(s_t)`.
- `λ = 1`: Results in the Monte Carlo advantage estimate `A = Q(s_t, a_t) - V(s_t)`.
A value of `λ` between 0 and 1 (e.g., 0.95-0.99) often provides the best performance.

### 6. Advantage Normalization
To stabilize training, it's common practice to standardize the estimated advantages in each batch to have a mean of 0 and a standard deviation of 1. This prevents large advantage values from causing destructive policy updates.

## Technical Implementation Details

### Files to Complete
You will implement the core logic in the following files:
-   **`cs285/agents/pg_agent.py`**:
    -   `estimate_advantage()`: Compute advantages using reward-to-go, baselines, and GAE.
    -   `train()`: Implement the policy update using the estimated advantages.
-   **`cs285/policies/MLP_policy.py`**:
    -   `update()`: Define the loss function and perform the gradient update for the policy network.
-   **`cs285/critics/bootstrapped_continuous_critic.py`**:
    -   Implement the value function network (`V(s)`) used as a baseline.

You will also need to copy over solutions from homework 1 for:
-   `infrastructure/rl_trainer.py`
-   `infrastructure/utils.py`
-   `policies/MLP_policy.py`

### Environments
This homework uses both classic control and MuJoCo environments:
-   **`CartPole-v0`**: A simple, low-dimensional environment for initial debugging.
-   **`LunarLander-v2`**: A more complex Box2D environment with discrete actions.
-   **`HalfCheetah-v4`**: A continuous control MuJoCo environment requiring more advanced techniques to solve.

## Structure

-   `cs285/`: Core RL codebase.
-   `cs285/agents/pg_agent.py`: Your main implementation file for the policy gradient agent.
-   `cs285/policies/MLP_policy.py`: The policy network.
-   `cs285/critics/bootstrapped_continuous_critic.py`: The baseline value network.
-   `cs285/scripts/run_hw2.py`: The script for running experiments.
-   `run_hw2.sh`: An automated script to run all experiment presets.
-   `README.md`: This file.

## Setup

1.  **Installation**: If you have not already, install MuJoCo and the required Python packages by following the instructions in `hw1/installation.md`.
2.  **Editable Install**: Install the `cs285` package in editable mode from the `hw2` directory:
    ```bash
    pip install -e .
    ```
3.  **Box2D**: Ensure you have the Box2D dependency for `LunarLander-v2`:
    ```bash
    pip install gym[box2d]
    ```

## Running Experiments

### Manual Execution
You can run individual experiments by calling `run_hw2.py` with different flags.

**Example (Small Lander Experiment)**:
```bash
python cs285/scripts/run_hw2.py 
    --env_name LunarLander-v2 
    --exp_name small_lander 
    --n_iter 100 
    --batch_size 1000 
    --learning_rate 0.01 
    --reward_to_go 
    --nn_baseline
```

**Example (HalfCheetah with GAE)**:
```bash
python cs285/scripts/run_hw2.py 
    --env_name HalfCheetah-v4 
    --exp_name cheetah_gae 
    --n_iter 150 
    --batch_size 5000 
    --learning_rate 0.02 
    --reward_to_go 
    --nn_baseline 
    --gae_lambda 0.97 
    --dont_standardize_advantages
```
Logs are stored inside `cs285/data/` by default.

### Automated Runner Script
To automate setup and run a sequence of policy gradient experiments, use the provided helper script:
```bash
chmod +x run_hw2.sh
./run_hw2.sh
```
This script will create a virtual environment, install dependencies, and run a suite of presets (vanilla PG, reward-to-go, baseline, etc.) on `LunarLander-v2`.

**Customize with Environment Variables**:
-   `ENV_NAME`: Target Gym environment (e.g., `CartPole-v0`, `HalfCheetah-v4`).
-   `PRESETS`: Comma-separated list of presets to run (e.g., `rtg,baseline,gae`).
-   `GAE_LAMBDA`: Lambda value for the `gae` preset.
-   `SKIP_INSTALL=1`: Reuse an existing environment.

**Example**: Run only reward-to-go with a baseline for `HalfCheetah-v4`.
```bash
PRESETS=rtg_baseline ENV_NAME=HalfCheetah-v4 ./run_hw2.sh
```

## Key Files

-   `cs285/agents/pg_agent.py`: Policy gradient agent implementation.
-   `cs285/policies/MLP_policy.py`: Policy network.
-   `cs285/critics/bootstrapped_continuous_critic.py`: Value network for the baseline.
-   `cs285/infrastructure/`: Core utilities and trainers.

## Submission

-   Submit your completed code, generated plots, and a report analyzing the results as specified in the assignment PDF.

## References

-   Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*.
-   Schulman, J., et al. (2015). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
-   Sutton & Barto, "Reinforcement Learning: An Introduction"
-   OpenAI Gymnasium documentation

## Running Experiments

- Example command:
  ```bash
  python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --exp_name pg_test
  ```
- Logs are stored inside `cs285/data/` by default.

## One-click runner script

You can automate setup and launch a sequence of policy-gradient experiments using the helper script:

```bash
chmod +x run_hw2.sh
./run_hw2.sh
```

The script will create (or reuse) a virtual environment in `.venv/`, install dependencies from `requirements.txt`, register the package in editable mode, and then execute a suite of presets (vanilla PG, reward-to-go, baseline, and reward-to-go with baseline) on `LunarLander-v2`.

Customize runs by overriding environment variables when invoking the script:

- `ENV_NAME` – target Gym environment (e.g. `CartPole-v1`, `LunarLander-v2`, `Ant-v4`).
- `PRESETS` – comma-separated list choosing among `vanilla`, `rtg`, `baseline`, `rtg_baseline`, `gae`, `no_std_adv`, or `custom`.
- `GAE_LAMBDA` – lambda for the `gae` preset (defaults to `0.95`).
- `N_ITER`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_AGENT_TRAIN_STEPS`, etc. – tune training hyperparameters.
- `SKIP_INSTALL=1` – reuse an existing environment without reinstalling requirements.
- `CUSTOM_FLAGS` – extra CLI flags appended to every run; required when using the `custom` preset.

Examples:

Run only reward-to-go with baseline for Hopper:

```bash
PRESETS=rtg_baseline ENV_NAME=Hopper-v4 ./run_hw2.sh
```

Evaluate a GAE sweep without reinstalling dependencies:

```bash
SKIP_INSTALL=1 PRESETS=gae ENV_NAME=HalfCheetah-v4 GAE_LAMBDA=0.97 ./run_hw2.sh
```

## Key Files

- `cs285/agents/`: Policy gradient agents
- `cs285/policies/`: Policy networks
- `cs285/critics/`: Value networks for baselines
- `cs285/infrastructure/`: Utilities and trainers

## Submission

- Submit code, results, and report as required.

## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. ICLR.
- Sutton & Barto, "Reinforcement Learning: An Introduction"
- OpenAI Gymnasium documentation
