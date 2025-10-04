# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 3: Q-Learning and Actor-Critic Methods

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment explores three fundamental paradigms in modern reinforcement learning: value-based methods (DQN), policy-based methods (Actor-Critic), and state-of-the-art algorithms that blend the two with entropy maximization (Soft Actor-Critic). You will implement these algorithms from the ground up, applying them to classic control and MuJoCo environments.

-   **Part 1: Deep Q-Networks (DQN)**. You will implement DQN to solve discrete action-space tasks like `LunarLander-v2`. This involves building a Q-network and using Experience Replay and Target Networks to stabilize training. You will also implement the Double DQN extension to mitigate value overestimation.

-   **Part 2: Actor-Critic (A2C)**. You will implement an Advantage Actor-Critic (A2C) agent for continuous control tasks. This involves training a policy network (the actor) and a value network (the critic) simultaneously, using the critic's value estimates to reduce the variance of the policy gradient.

-   **Part 3: Soft Actor-Critic (SAC)**. You will implement SAC, an advanced off-policy, maximum-entropy actor-critic algorithm. SAC is known for its sample efficiency and stability, achieving excellent performance on complex continuous control problems.

## Learning Objectives

-   Implement and understand **Deep Q-Networks (DQN)** for discrete control.
-   Master the use of **Experience Replay** and **Target Networks** for stabilizing off-policy training.
-   Implement an **Advantage Actor-Critic (A2C)** agent for continuous control.
-   Understand the roles of the **actor** (policy) and **critic** (value function) and how they collaborate.
-   Implement **Soft Actor-Critic (SAC)**, a state-of-the-art maximum entropy algorithm.
-   Analyze the trade-offs between on-policy (A2C) and off-policy (DQN, SAC) learning.

## Key Concepts

### 1. Deep Q-Learning (DQN)

DQN is a value-based, off-policy algorithm that learns an optimal action-value function, `Q*(s, a)`. It uses a deep neural network to approximate `Q(s, a; θ)`.

-   **Bellman Equation**: The core of Q-learning is the Bellman equation, which defines the optimal Q-value as the expected immediate reward plus the discounted maximum Q-value of the next state:
    $$Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)} [r(s, a) + \gamma \max_{a'} Q^*(s', a')]$$

-   **Loss Function**: The network is trained by minimizing the mean squared error between the predicted Q-value and a target Q-value derived from the Bellman equation:
    $$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q(s, a; \theta) - y \right)^2 \right]$$
    where the target `y` is:
    $$y = r + \gamma \max_{a'} Q(s', a'; \theta_{\text{target}})$$

-   **Experience Replay**: To break data correlations and improve sample efficiency, transitions `(s, a, r, s', done)` are stored in a large replay buffer. The network is trained on mini-batches randomly sampled from this buffer.

-   **Target Network**: To stabilize training, a separate "target network" `Q(s', a'; θ_target)` is used to compute the target values `y`. The target network's weights are periodically and slowly updated with the online network's weights (`θ_target ← τ * θ + (1-τ) * θ_target`), which prevents the optimization target from changing at every step.

-   **Double DQN (DDQN)**: Standard DQN is prone to overestimating Q-values because the `max` operator uses the same network to both select and evaluate an action. DDQN decouples this by using the online network to select the best action and the target network to evaluate it:
    $$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta_{\text{target}})$$

### 2. Actor-Critic (A2C)

Actor-Critic methods combine the strengths of policy-based and value-based methods. They consist of two components:
-   **Actor**: A policy network `π(a|s; θ)` that learns to select actions.
-   **Critic**: A value network `V(s; φ)` that estimates the value of being in a state `s`.

The actor updates its policy in the direction suggested by the critic. The critic, in turn, evaluates the actor's new policy.

-   **Advantage Function**: Instead of using the raw return, A2C uses the **advantage function** `A(s, a)` to reduce the variance of the policy gradient. The advantage indicates how much better an action `a` is compared to the average action from state `s`. It is estimated as:
    $$A(s, a) = Q(s, a) - V(s) \approx (r + \gamma V(s')) - V(s)$$
    The term `(r + \gamma V(s')) - V(s)` is the **TD Error**, which serves as a low-variance estimate of the advantage.

-   **Policy (Actor) Update**: The actor's weights `θ` are updated using the policy gradient theorem, with the advantage estimate `A` providing the learning signal:
    $$\nabla_{\theta} J(\theta) \approx \mathbb{E} [\nabla_{\theta} \log \pi(a|s; \theta) A(s, a)]$$

-   **Value (Critic) Update**: The critic's weights `φ` are updated by minimizing the squared TD error, making its value estimates more accurate:
    $$L(\phi) = \mathbb{E} [((r + \gamma V(s'; \phi)) - V(s; \phi))^2]$$

### 3. Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm based on the **maximum entropy reinforcement learning** framework. The goal is to learn a policy that not only maximizes the expected cumulative reward but also the policy's entropy. This encourages exploration and leads to more robust and stable policies.

-   **Maximum Entropy Objective**:
    $$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_{\pi}} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$
    Here, `α` is a temperature parameter that controls the importance of the entropy term `H` relative to the reward.

-   **Key Components**: SAC uses several networks:
    1.  **Stochastic Policy (Actor)**: `π(a|s; θ)`.
    2.  **Two Soft Q-Value Networks (Critics)**: `Q(s, a; φ_1)` and `Q(s, a; φ_2)`. Using two Q-networks (a "clipped double-Q" trick) helps mitigate the positive bias in value estimates. The minimum of the two is used for policy updates.
    3.  **Two Target Q-Networks**: Target networks are used to stabilize the soft Bellman backup, similar to DQN.

-   **Updates**:
    -   **Critic Update**: The Q-networks are trained to minimize the soft Bellman residual:
        $$L(\phi_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q(s, a; \phi_i) - (r + \gamma (\min_{j=1,2} Q(s', a'; \phi_{\text{target},j}) - \alpha \log \pi(a'|s'; \theta))) \right)^2 \right]$$
        where `a'` is sampled from the current policy `π(·|s')`.
    -   **Actor Update**: The policy is updated to maximize the expected future Q-value and entropy:
        $$\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} [\nabla_{\theta} \log \pi(a_{\theta}(s, \epsilon)|s) (\alpha \log \pi(a_{\theta}(s, \epsilon)|s) - \min_{j=1,2} Q(s, a_{\theta}(s, \epsilon); \phi_j))]$$
    -   **Temperature Update**: The temperature `α` can be automatically tuned to balance the reward and entropy terms.

## Technical Implementation Details

### Files to Complete

You will implement the core logic in the following files.

**Part 1: DQN**
-   **`cs285/agents/dqn_agent.py`**:
    -   `train()`: Implement the DQN training loop, including sampling from the replay buffer and computing the TD loss.
    -   `add_to_replay_buffer()`: Add new experiences to the replay buffer.
-   **`cs285/critics/dqn_critic.py`**:
    -   `update()`: Compute the TD target and the loss for the Q-network.
-   **`cs285/policies/argmax_policy.py`**:
    -   `get_action()`: Implement the epsilon-greedy exploration strategy.

**Part 2: Actor-Critic**
-   **`cs285/agents/ac_agent.py`**:
    -   `train()`: Implement the A2C training loop.
    -   `estimate_advantage()`: Calculate advantage estimates using the critic's value function.
-   **`cs285/critics/bootstrapped_continuous_critic.py`**:
    -   `forward()`: Define the value network architecture.
    -   `update()`: Implement the critic's training step.
-   **`cs285/policies/MLP_policy.py`**:
    -   `update()`: Implement the actor's policy gradient update using advantage estimates.

**Part 3: Soft Actor-Critic**
-   **`cs285/agents/sac_agent.py`**:
    -   Implement the main training logic for the SAC agent.
-   **`cs285/critics/sac_critic.py`**:
    -   Implement the soft Q-value critic, including the Bellman backup.
-   **`cs285/policies/sac_policy.py`**:
    -   Implement the stochastic actor and its update rule.

You will also need to copy over solutions from previous homeworks for:
-   `infrastructure/rl_trainer.py`
-   `infrastructure/utils.py`
-   `policies/MLP_policy.py` (for A2C)

### Environments
-   **`LunarLander-v2`**: A discrete action space environment for DQN.
-   **`CartPole-v0`**: A simple continuous control environment for A2C and SAC.
-   **`HalfCheetah-v4`**: A more complex MuJoCo environment for advanced continuous control.

## Structure

-   `cs285/`: Core RL codebase.
-   `cs285/agents/`: Contains `dqn_agent.py`, `ac_agent.py`, and `sac_agent.py`.
-   `cs285/critics/`: Contains `dqn_critic.py`, `bootstrapped_continuous_critic.py`, and `sac_critic.py`.
-   `cs285/policies/`: Contains `argmax_policy.py` (for DQN) and policy networks for actor-critic methods.
-   `cs285/scripts/`: Contains `run_hw3_dqn.py`, `run_hw3_actor_critic.py`, and `run_hw3_sac.py`.
-   `run_hw3.sh`: An automated script to run all three parts.
-   `README.md`: This file.

## Setup

1.  **Installation**: If you have not already, install MuJoCo and the required Python packages by following the instructions in `hw1/installation.md`.
2.  **Editable Install**: Install the `cs285` package in editable mode from the `hw3` directory:
    ```bash
    pip install -e .
    ```

## Running Experiments

### Manual Execution
You can run individual experiments for each part of the assignment.

**Example (DQN on LunarLander)**:
```bash
python cs285/scripts/run_hw3_dqn.py 
    --env_name LunarLander-v2 
    --exp_name dqn_lander
```

**Example (Actor-Critic on CartPole)**:
```bash
python cs285/scripts/run_hw3_actor_critic.py 
    --env_name CartPole-v0 
    --exp_name ac_cartpole 
    --batch_size 1000 
    --n_iter 100
```

**Example (SAC on HalfCheetah)**:
```bash
python cs285/scripts/run_hw3_sac.py 
    --env_name HalfCheetah-v4 
    --exp_name sac_cheetah
```
Logs are stored inside `cs285/data/` by default.

### Automated Runner Script
To automate the setup and execution of all three parts (DQN, Actor-Critic, and SAC), use the provided helper script:
```bash
chmod +x run_hw3.sh
./run_hw3.sh
```
This script will create a virtual environment, install dependencies, and run the default experiments for each algorithm.

**Customize with Environment Variables**:
-   `RUN_DQN=0`, `RUN_ACTOR_CRITIC=0`, `RUN_SAC=0`: Set to `0` to skip a specific part.
-   `DQN_ENV_NAME`, `AC_ENV_NAME`, `SAC_ENV_NAME`: Specify different environments for each algorithm.
-   `DQN_DOUBLE_Q=1`: Enable Double DQN.
-   `AC_STANDARDIZE_ADV=0`: Disable advantage standardization in A2C.
-   `SAC_INIT_TEMPERATURE`: Set the initial temperature for SAC.
-   `SKIP_INSTALL=1`: Reuse an existing environment.

**Example**: Run only Double DQN on `Pong` and SAC on `HalfCheetah`.
```bash
RUN_ACTOR_CRITIC=0 
DQN_ENV_NAME=PongNoFrameskip-v4 
DQN_DOUBLE_Q=1 
SAC_ENV_NAME=HalfCheetah-v4 
./run_hw3.sh
```

## Key Files

-   `cs285/agents/dqn_agent.py`: DQN agent implementation.
-   `cs285/agents/ac_agent.py`: Actor-Critic agent implementation.
-   `cs285/agents/sac_agent.py`: Soft Actor-Critic agent implementation.
-   `cs285/critics/`: Critic network implementations for each algorithm.
-   `cs285/policies/`: Policy implementations.

## Submission

-   Submit your completed code, generated plots, and a report analyzing the results as specified in the assignment PDF.

## References

-   Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
-   Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv*.
-   Haarnoja, T., et al. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*.
-   OpenAI Gymnasium documentation

## Key Files

- `cs285/agents/`: DQN, Actor-Critic, SAC agents
- `cs285/policies/`: Policy networks
- `cs285/critics/`: Q-networks and value networks

## Submission

- Submit code, results, and report as required.

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv.
- Haarnoja, T., et al. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. ICML.
- OpenAI Gymnasium documentation

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from hw1 or hw2`.

- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

You will then need to implement new routines in the following files for homework 3 part 1 (Q-learning):

- [agents/dqn_agent.py](cs285/agents/dqn_agent.py)
- [critics/dqn_critic.py](cs285/critics/dqn_critic.py)
- [policies/argmax_policy.py](cs285/policies/argmax_policy.py)

and in the following files for part 2 (actor-critic):

- [agents/ac_agent.py](cs285/agents/ac_agent.py)
- [critics/bootstrapped_continuous_critic.py](cs285/critics/bootstrapped_continuous_critic.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)

The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw3_dqn.py](cs285/scripts/run_hw3_dqn.py) and [scripts/run_hw3_actor_critic](cs285/scripts/run_hw3_actor_critic.py), though you will not need to edit this files beyond changing runtime arguments.

See the [assignment PDF](cs285_hw3.pdf) for more details on what files to edit.
