# Author: Taha Majlesi - 810101504, University of Tehran
# Homework 3: Q-Learning and Actor-Critic Methods

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment covers value-based and actor-critic reinforcement learning algorithms. You will implement Deep Q-Networks (DQN) for discrete action spaces, explore experience replay and target networks, and develop actor-critic methods for continuous control. The assignment is divided into three parts: Q-learning, actor-critic, and soft actor-critic.

## Learning Objectives

- Master value-based RL with Q-learning and DQN
- Understand experience replay and target networks for stable training
- Implement actor-critic algorithms for continuous action spaces
- Analyze the differences between on-policy and off-policy methods
- Explore entropy regularization in SAC

## Key Concepts

- **Q-Learning**: Off-policy TD learning for optimal Q-function
- **Deep Q-Networks (DQN)**: Using neural networks for Q-function approximation
- **Experience Replay**: Storing and sampling past experiences for efficient learning
- **Target Networks**: Stabilizing training by slowly updating target Q-networks
- **Actor-Critic**: Combining policy (actor) and value (critic) function learning
- **Advantage Actor-Critic (A2C)**: Synchronous actor-critic with multiple workers
- **Soft Actor-Critic (SAC)**: Maximum entropy RL for exploration and stability

## Structure

- `cs285/`: RL codebase (agents, policies, critics)
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

- Example command:
  ```bash
  python cs285/scripts/run_hw3.py --env_name LunarLander-v2 --exp_name ac_test
  ```
- Logs produced by the helper scripts land in `cs285/data/` by default.

## Automated runner script

To set up the environment and execute the DQN, actor-critic, and SAC experiments in one pass, use the included helper script:

```bash
chmod +x run_hw3.sh
./run_hw3.sh
```

The script will create (or reuse) a virtual environment at `.venv/`, install dependencies from `requirements.txt`, install `cs285` in editable mode, and then sequentially launch the three training pipelines with sensible default hyperparameters (`LunarLander-v3` for DQN, `CartPole-v0` for actor-critic and SAC).

Customize the workflow by overriding environment variables when invoking the script:

- `RUN_DQN`, `RUN_ACTOR_CRITIC`, `RUN_SAC` – set to `0` to skip a section.
- `DQN_ENV_NAME`, `AC_ENV_NAME`, `SAC_ENV_NAME` – target environments supported by the respective scripts.
- `DQN_DOUBLE_Q=1` – enable Double DQN.
- `AC_STANDARDIZE_ADV=0` – disable advantage standardization.
- `SAC_INIT_TEMPERATURE`, `SAC_ACTOR_UPDATE_FREQ`, etc. – fine-tune SAC hyperparameters.
- `SKIP_INSTALL=1` – reuse an existing virtualenv without reinstalling packages.
- `*_EXTRA_FLAGS` (e.g. `DQN_EXTRA_FLAGS="--save_params"`) – append additional CLI options to a stage.

Examples:

Run only Double DQN on Pong with a cold start:

```bash
RUN_ACTOR_CRITIC=0 RUN_SAC=0 DQN_ENV_NAME=PongNoFrameskip-v4 DQN_DOUBLE_Q=1 ./run_hw3.sh
```

Reuse the environment to launch SAC on HalfCheetah with a custom temperature:

```bash
SKIP_INSTALL=1 RUN_DQN=0 RUN_ACTOR_CRITIC=0 SAC_ENV_NAME=HalfCheetah-v4 SAC_INIT_TEMPERATURE=0.5 ./run_hw3.sh
```

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
