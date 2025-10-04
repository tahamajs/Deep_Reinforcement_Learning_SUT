# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 4: Model-Based Reinforcement Learning

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment delves into **Model-Based Reinforcement Learning (MBRL)**, a paradigm focused on improving sample efficiency by learning a model of the environment's dynamics. Instead of interacting with the real world for every learning step (as in model-free RL), an MBRL agent learns a predictive model of how the world behaves. This "world model" can then be used as a simulator to plan actions, generate synthetic experience, and train a policy with far fewer real-world interactions.

You will implement two major MBRL approaches:

-   **Part 1: Model Predictive Control (MPC)**. You will first build a neural network that learns the environment's dynamics: `s_t+1, r_t = f(s_t, a_t)`. Then, you will implement an MPC controller that uses this model to "look ahead" and choose the best action at each step by optimizing a sequence of future actions. This involves sampling action sequences, predicting their outcomes with the learned model, and selecting the best one.

-   **Part 2: Model-Based Policy Optimization (MBPO)**. You will implement MBPO, a more advanced algorithm that combines the benefits of model-based planning and model-free policy learning. MBPO uses the learned dynamics model to generate "rollouts"—short, simulated trajectories that start from real states. These synthetic rollouts are then added to the replay buffer of a powerful model-free algorithm (like Soft Actor-Critic), allowing the policy to be trained on a mix of real and imagined data, dramatically improving sample efficiency.

## Learning Objectives

-   Understand the core principles of **Model-Based RL** and its advantages in sample efficiency.
-   Implement a **neural network dynamics model** to predict state transitions and rewards.
-   Develop a **Model Predictive Control (MPC)** agent that uses the learned model for online planning.
-   Implement and understand action sequence optimization methods like **Random Shooting** and the **Cross-Entropy Method (CEM)**.
-   Implement **Model-Based Policy Optimization (MBPO)**, a hybrid algorithm that uses model-generated data to train a model-free policy.
-   Analyze the impact of model error and how techniques like short rollouts can mitigate it.

## Key Concepts

### 1. Dynamics Model

The foundation of MBRL is the dynamics model, `f_φ(s_t, a_t)`, which is a supervised learning model trained to predict the next state and reward given the current state and action.
$$ \hat{s}_{t+1}, \hat{r}_t = f_{\phi}(s_t, a_t) $$
This model is typically a feedforward neural network trained on a dataset `D = {(s, a, r, s')}` of real transitions collected from the environment. The loss function is the Mean Squared Error (MSE) between the model's predictions and the true outcomes.
$$ L(\phi) = \sum_{(s, a, r, s') \in \mathcal{D}} ||f_{\phi}(s, a) - (s', r)||^2 $$
To handle stochastic environments and model uncertainty, an ensemble of dynamics models can be used.

### 2. Model Predictive Control (MPC)

MPC is an online planning algorithm that uses the learned dynamics model to select the best action at each time step. It works as follows:

1.  **Observe** the current state `s_t`.
2.  **Plan** a sequence of future actions `(a_t, a_{t+1}, ..., a_{t+H-1})` by optimizing a reward objective *inside the learned model*. This planning step involves:
    a.  Generating a set of candidate action sequences.
    b.  For each sequence, using the dynamics model `f_φ` to predict the resulting trajectory of states and rewards: `(s_{t+1}, r_t), (s_{t+2}, r_{t+1}), ...`.
    c.  Evaluating each trajectory by summing the predicted rewards: `Σ r_i`.
    d.  Selecting the action sequence that yields the highest total reward.
3.  **Execute** only the *first* action `a_t` from the best sequence in the real environment.
4.  **Discard** the rest of the plan and repeat the process from the new state `s_{t+1}`.

This "plan, execute one step, replan" cycle makes MPC robust to model errors, as it constantly corrects its plan based on real-world feedback.

-   **Action Sequence Sampling**:
    -   **Random Shooting**: The simplest method, where `N` random action sequences are sampled and the best one is chosen.
    -   **Cross-Entropy Method (CEM)**: An iterative optimization technique that refines the search for the best action sequence. It starts with a distribution over sequences (e.g., a Gaussian), selects the top "elite" sequences, and fits a new distribution to these elites. This process is repeated for several iterations to converge on a high-reward action sequence.

### 3. Model-Based Policy Optimization (MBPO)

MBPO is a hybrid algorithm that leverages a learned model to augment the training data for a model-free RL algorithm, typically an off-policy one like SAC. This approach combines the planning capabilities of a model with the asymptotic performance of a model-free policy.

The MBPO loop is as follows:

1.  **Collect** a small amount of real data from the environment using the current policy. Add this to a replay buffer `D_real`.
2.  **Train** the dynamics model `f_φ` on the real data in `D_real`.
3.  **Generate Synthetic Data**:
    a.  Sample a state `s_t` from the real replay buffer `D_real`.
    b.  Perform a `k`-step rollout using the learned model. At each step `i` in the rollout:
        i.  The current policy `π` selects an action `a_i = π(s_i)`.
        ii. The dynamics model predicts the next state `s_{i+1} = f_φ(s_i, a_i)`.
    c.  Add these `k` synthetic transitions `(s_i, a_i, r_i, s_{i+1})` to a separate model-based replay buffer `D_model`.
4.  **Train Policy**: Update the model-free agent (e.g., SAC) by sampling mini-batches from a mix of real data (`D_real`) and model-generated data (`D_model`).

By using **short rollouts** (small `k`), MBPO mitigates the problem of compounding model error, where small prediction errors accumulate over long trajectories, leading to unrealistic and unhelpful synthetic data. This allows the policy to benefit from imagined experience without being led too far astray by an imperfect model.

## Structure

- `cs285/`: RL codebase (model-based agents, dynamics models, MPC policies)
- `results/`: Experiment outputs
- `README.md`: This file
- `*.ipynb`: Notebooks for analysis

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Use a virtual environment.

## Running Experiments

- Model-based MPC (Question 1):
  ```bash
  python cs285/scripts/run_hw4_mb.py --env_name cheetah-cs285-v0 --exp_name mb_test
  ```
- MBPO (Question 2):
  ```bash
  python cs285/scripts/run_hw4_mbpo.py --env_name cheetah-cs285-v0 --exp_name mbpo_test
  ```
- Logs generated by these scripts are stored in `cs285/data/` by default.

## Automated runner script

To bootstrap a virtual environment and execute the default MPC and MBPO experiments in one step, use the helper script:

```bash
chmod +x run_hw4.sh
./run_hw4.sh
```

The script creates (or reuses) `.venv/`, installs dependencies from `requirements.txt`, installs `cs285` in editable mode, and then launches both experiments with MuJoCo configured for headless rendering (`MUJOCO_GL=egl`).

Customize behavior with environment variables when invoking the script:

- `RUN_MB`, `RUN_MBPO` – set to `0` to skip either stage.
- `MB_ENV_NAME`, `MBPO_ENV_NAME` – choose among the provided CS285 benchmark environments.
- `MB_MPC_SAMPLING` – toggle between `random` and `cem`; combine with `MB_CEM_*` parameters to configure CEM.
- `MB_ADD_SL_NOISE=1` or `MBPO_ADD_SL_NOISE=1` – inject supervised-learning noise during dynamics training.
- `SAC_*` variables – adjust the SAC hyperparameters used inside MBPO (e.g. `SAC_INIT_TEMPERATURE=0.5`).
- `SKIP_INSTALL=1` – reuse an existing environment without reinstalling dependencies.
- `MB_EXTRA_FLAGS`, `MBPO_EXTRA_FLAGS` – append arbitrary CLI flags to the respective Python commands.

Examples:

Run only CEM-based MPC on the obstacles environment:

```bash
RUN_MBPO=0 MB_ENV_NAME=obstacles-cs285-v0 MB_MPC_SAMPLING=cem MB_CEM_ITERATIONS=6 MB_CEM_NUM_ELITES=10 ./run_hw4.sh
```

Reuse the environment and launch MBPO with a shorter rollout length:

```bash
SKIP_INSTALL=1 RUN_MB=0 MBPO_ENV_NAME=cheetah-cs285-v0 MBPO_ROLLOUT_LENGTH=5 ./run_hw4.sh
```

## Key Files

- `cs285/agents/`: Model-based agents (MB, MBPO)
- `cs285/models/`: Dynamics models
- `cs285/policies/`: MPC policies
- `cs285/infrastructure/`: Utilities and trainers

## Submission

- Submit code, results, and report as required.

## References

- Deisenroth, M., & Rasmussen, C. E. (2011). PILCO: A model-based and data-efficient approach to policy search. ICML.
- Janner, M., et al. (2019). When to trust your model: Model-based policy optimization. NeurIPS.
- Chua, K., et al. (2018). Deep reinforcement learning in a handful of trials using probabilistic dynamics models. NeurIPS.
- OpenAI Gymnasium documentation

## Complete the code

The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from Piazza'.

- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](cs285/infrastructure/utils.py)

You will then need to implement code in the following files:

- [agents/mb_agent.py](cs285/agents/mb_agent.py)
- [models/ff_model.py](cs285/models/ff_model.py)
- [policies/MPC_policy.py](cs285/policies/MPC_policy.py)
- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [agents/mbpo_agent.py](cs285/infrastructure/rl_trainer.py)

The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw4_mb.py](cs285/scripts/run_hw4_mb.py) and [scripts/run_hw4_mbpo.py](cs285/scripts/run_hw4_mbpo.py), though you will not need to edit this files beyond changing runtime arguments.

See the [assignment PDF](cs285_hw4.pdf) for more details on what files to edit.
