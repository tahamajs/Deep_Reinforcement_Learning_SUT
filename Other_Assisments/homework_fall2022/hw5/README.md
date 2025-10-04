# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 5: Offline Reinforcement Learning and Exploration

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment explores the frontier of reinforcement learning, focusing on two critical areas: **Offline RL** and **Exploration**. You will move beyond traditional online learning and tackle the challenge of learning effective policies from static, pre-collected datasets. This is crucial for real-world applications where active data collection is expensive, risky, or impractical.

The assignment is structured into three main parts:

-   **Part 1: Exploration with Random Network Distillation (RND)**. You will first implement RND, a powerful curiosity-driven exploration technique. RND provides an "intrinsic reward" to the agent for visiting novel states, encouraging it to explore its environment thoroughly even when extrinsic rewards are sparse. You will combine this with an offline RL algorithm (CQL) to create an agent that can both explore and exploit effectively.

-   **Part 2: Offline RL with Advantage-Weighted Actor-Critic (AWAC)**. You will implement AWAC, an offline actor-critic algorithm that leverages the advantages of actions in the dataset to guide policy learning. It's a simple yet effective method that constrains the learned policy to stay close to the behavior policy that generated the data, mitigating distribution shift.

-   **Part 3: Offline RL with Implicit Q-Learning (IQL)**. You will implement IQL, a powerful offline RL algorithm that avoids the pitfalls of explicit policy constraints and Q-value maximization. Instead, it learns a Q-function implicitly via expectile regression and extracts a policy from it. This approach has proven to be highly effective and stable for offline learning.

## Learning Objectives

-   Understand the fundamental challenges of **Offline Reinforcement Learning**, particularly **distributional shift**.
-   Implement and analyze three distinct offline RL algorithms: **AWAC**, **IQL**, and **CQL**.
-   Master **Random Network Distillation (RND)** as a technique for intrinsic motivation and curiosity-driven exploration.
-   Learn how to combine exploration bonuses with exploitation policies.
-   Implement **Conservative Q-Learning (CQL)**, which combats value overestimation in offline settings.
-   Understand how **Advantage-Weighted Actor-Critic (AWAC)** uses implicit constraints for stable offline policy updates.
-   Implement **Implicit Q-Learning (IQL)** using expectile regression to learn value functions without explicit Bellman backups on out-of-distribution actions.

## Key Concepts

### 1. Offline Reinforcement Learning

Offline RL (also known as batch RL) is a paradigm where the agent learns from a fixed dataset `D` of transitions `(s, a, r, s')` collected by some unknown "behavior policy" `π_β`. The agent has **no access** to the live environment for further interaction.

-   **The Challenge of Distributional Shift**: The primary obstacle in offline RL is distributional shift. Standard off-policy algorithms like Q-learning fail because they try to evaluate actions that are not present in the dataset. This leads to bootstrapping from erroneously high Q-values for out-of-distribution (OOD) actions, causing value overestimation and policy divergence.
-   **The Goal**: The goal of offline RL is to learn the best possible policy that can be extracted from the given dataset, while avoiding the temptation to query actions that the dataset cannot provide information about. This is often achieved by constraining the learned policy to stay "close" to the behavior policy.

### 2. Random Network Distillation (RND)

RND is an exploration technique that provides the agent with an **intrinsic reward** for visiting novel states. It consists of two neural networks:
1.  A **Target Network**: A randomly initialized network that is frozen. It takes a state `s` and produces a fixed, random embedding `f(s)`.
2.  A **Predictor Network**: A trainable network that tries to predict the output of the target network: `f̂(s; θ)`.

-   **Intrinsic Reward**: The prediction error `||f̂(s; θ) - f(s)||^2` serves as the intrinsic reward.
    -   When the agent visits a state `s` for the first time, the predictor network will have a high error, generating a large intrinsic reward and encouraging the agent to explore that state.
    -   As the agent visits the state more often, the predictor network learns to accurately predict the target network's output, and the intrinsic reward diminishes.
-   **Total Reward**: The agent is trained on a combination of the extrinsic (environment) reward and the intrinsic (curiosity) reward: `r_total = r_extrinsic + β * r_intrinsic`.

### 3. Conservative Q-Learning (CQL)

CQL is an offline RL algorithm that directly addresses the problem of Q-value overestimation for OOD actions. It adds a regularization term to the standard Bellman error objective.

-   **The CQL Objective**: The key idea is to simultaneously **minimize** the Q-values for actions chosen by the policy while **maximizing** the Q-values for actions that were actually present in the dataset.
    $$ L_{\text{CQL}}(\theta) = \alpha \left( \mathbb{E}_{s \sim \mathcal{D}}[\log\sum_a \exp(Q(s,a;\theta))] - \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a;\theta)] \right) + L_{\text{Bellman}}(\theta) $$
    -   The first term pushes down the values of all actions in a state (sampled from the policy's distribution).
    -   The second term pushes up the values of actions from the dataset.
-   **Effect**: This conservative objective ensures that the Q-values for unseen actions are not arbitrarily high, forcing the policy to prefer actions that are well-supported by the data.

### 4. Advantage-Weighted Actor-Critic (AWAC)

AWAC is a simple and effective offline actor-critic algorithm. It constrains the policy to stay close to the behavior policy by weighting the policy update by the advantage of the actions in the dataset.

-   **Policy Update**: The policy is updated via supervised learning on the actions from the dataset, but each action is weighted by the exponentiated advantage:
    $$ \theta \leftarrow \arg\max_{\theta} \mathbb{E}_{(s,a) \sim \mathcal{D}}[\log \pi_{\theta}(a|s) \exp(\frac{1}{\lambda} A^{\pi_k}(s,a))] $$
    where `A(s,a) = Q(s,a) - V(s)` is the advantage, and `λ` is a temperature parameter.
-   **Intuition**:
    -   Actions with high advantages (`A > 0`) receive a large weight, encouraging the policy to imitate them.
    -   Actions with low advantages (`A < 0`) receive a small weight, discouraging the policy from imitating them.
-   This acts as an implicit constraint, keeping the policy from deviating too far from the good parts of the behavior policy.

### 5. Implicit Q-Learning (IQL)

IQL is a powerful offline algorithm that avoids directly maximizing Q-values over OOD actions. Instead, it learns a Q-function and extracts a policy from it in a more robust way.

-   **Key Ideas**:
    1.  **Expectile Regression**: Instead of using the `max` operator in the Bellman backup (which causes OOD issues), IQL uses **expectile regression** to estimate the value function `V`. This provides an upper-envelope of the state's value without explicitly querying OOD actions.
    2.  **Advantage-Weighted Regression**: The policy is then trained via supervised learning, similar to AWAC, by imitating actions from the dataset weighted by their exponentiated Q-values (which serve as an estimate of the advantage).
-   **IQL Critic Update**: The Q-function is learned by minimizing a standard Bellman error, but the target uses the expectile-based value function `V`.
-   **IQL Actor Update**: The policy is trained to maximize `E_{(s,a)~D}[exp(β(Q(s,a) - V(s))) log π(a|s)]`.
-   **Result**: IQL learns a well-behaved Q-function and can extract a policy that significantly improves upon the behavior policy, making it one of the top-performing offline RL algorithms.

## Structure

- `cs285/`: RL codebase (AWAC, CQL, IQL, RND, infrastructure)
- `results/`: Experiment outputs
- `README.md`: This file
- `*.ipynb`: Notebooks for analysis

## Setup

You can run this code on your own machine.

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. There are two new package requirements (`opencv-python` and `gym[atari]`) beyond what was used in the previous assignments; make sure to install these with `pip install -r requirements.txt` if you are running the assignment locally.

## Complete the code

The following files have blanks to be filled with your solutions from homework 1 and 3. The relevant sections are marked with `TODO'. You can get solutions from Ed.

- [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
- [policies/argmax_policy.py](cs285/policies/argmax_policy.py)
- [critics/dqn_critic.py](cs285/critics/dqn_critic.py)

You will then need to implement code in the following files:

For RND + CQL:

- [exploration/rnd_model.py](cs285/exploration/rnd_model.py)
- [agents/explore_or_exploit_agent.py](cs285/agents/explore_or_exploit_agent.py)
- [critics/cql_critic.py](cs285/critics/cql_critic.py)

For AWAC:

- [agents/awac_agent.py](cs285/agents/awac_agent.py)

For IQL:

- [agents/iql_agent.py](cs285/agents/iql_agent.py)
- [critics/iql_critic.py](cs285/critics/iql_critic.py)

The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw5_expl.py](cs285/scripts/run_hw5_expl.py), though you will not need to edit this files beyond changing runtime arguments.

See the [assignment PDF](hw5.pdf) for more details on what files to edit.

For this particular assignment, you will need to install networkx==2.5

## Running Experiments

- Exploration + CQL (Question 1):
  ```bash
  python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --exp_name expl_test
  ```
- AWAC (Question 2):
  ```bash
  python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name awac_test
  ```
- IQL (Question 3):
  ```bash
  python cs285/scripts/run_hw5_iql.py --env_name PointmassHard-v0 --exp_name iql_test
  ```
- Each script writes logs to `cs285/data/`.

## Automated runner script

To provision a virtual environment, install dependencies (including `networkx==2.5`), and execute the three homework pipelines with one command, use the helper script:

```bash
chmod +x run_hw5.sh
./run_hw5.sh
```

By default it runs the exploration/CQL, AWAC, and IQL experiments sequentially on the pointmass benchmarks with RND enabled for intrinsic rewards. The script creates (or reuses) `.venv/`, installs packages from `requirements.txt`, registers `cs285` in editable mode, and sets `MUJOCO_GL=egl` for headless rendering.

Tweak behavior by setting environment variables when launching the script:

- `RUN_EXPLORATION`, `RUN_AWAC`, `RUN_IQL` – set to `0` to skip a stage.
- `EXPL_ENV_NAME`, `AWAC_ENV_NAME`, `IQL_ENV_NAME` – choose among `PointmassEasy/Medium/Hard/VeryHard-v0`.
- `EXPL_USE_RND`, `AWAC_USE_RND`, `IQL_USE_RND` – toggle Random Network Distillation bonuses.
- `AWAC_LAMBDA`, `IQL_EXPECTILE` – adjust algorithm-specific hyperparameters.
- `EXPL_EXTRA_FLAGS`, `AWAC_EXTRA_FLAGS`, `IQL_EXTRA_FLAGS` – append custom CLI arguments.
- `SKIP_INSTALL=1` – reuse the existing environment without reinstalling packages.

Examples:

Run only AWAC with supervised data and no intrinsic rewards:

```bash
RUN_EXPLORATION=0 RUN_IQL=0 AWAC_USE_RND=0 AWAC_ENV_NAME=PointmassMedium-v0 ./run_hw5.sh
```

Reuse the environment and sweep IQL expectile on the hard task:

```bash
SKIP_INSTALL=1 RUN_EXPLORATION=0 RUN_AWAC=0 IQL_ENV_NAME=PointmassHard-v0 IQL_EXPECTILE=0.9 ./run_hw5.sh
```

## Key Files

- `cs285/agents/`: AWAC, CQL, IQL agents
- `cs285/critics/`: CQL, IQL critics
- `cs285/exploration/`: RND exploration model
- `cs285/policies/`: Policy networks
- `cs285/infrastructure/`: Utilities and trainers

## Submission

- Submit code, results, and report as required.

## References

- Kumar, A., et al. (2020). Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS.
- Nair, A., et al. (2020). AWAC: Accelerating Online Reinforcement Learning with Offline Datasets. arXiv.
- Kostrikov, I., et al. (2022). Offline Reinforcement Learning with Implicit Q-Learning. ICLR.
- Burda, Y., et al. (2019). Exploration by Random Network Distillation. ICLR.
- OpenAI Gymnasium documentation
