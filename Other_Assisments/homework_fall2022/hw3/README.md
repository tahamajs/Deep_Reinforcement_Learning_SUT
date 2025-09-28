# Homework 3: Q-Learning and Actor-Critic Methods

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
- Results are saved in `results/`.

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

You may also want to look through [scripts/run_hw3_dqn.py](cs285/scripts/run_hw3_dqn.py) and [scripts/run_hw3_actor_critic](cs285/scripts/run_hw3_actor_critic.py) (if running locally) or [scripts/run_hw3.ipynb](cs285/scripts/run_hw3.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

See the [assignment PDF](cs285_hw3.pdf) for more details on what files to edit.
