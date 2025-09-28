# Homework 5: Offline and Advanced Deep Reinforcement Learning

**Author:** Taha Majlesi - 810101504, University of Tehran

## Overview

This assignment focuses on offline reinforcement learning and advanced exploration techniques. You will implement and analyze algorithms that learn from fixed datasets without environment interaction, including Conservative Q-Learning (CQL), Advantage-Weighted Actor-Critic (AWAC), and Implicit Q-Learning (IQL). Additionally, you'll explore Random Network Distillation (RND) for intrinsic motivation and curiosity-driven exploration.

## Learning Objectives

- Master offline RL algorithms that work with static datasets
- Understand conservative and advantage-weighted policy learning
- Implement exploration bonuses using intrinsic motivation
- Analyze the challenges of offline RL (distribution shift, bootstrapping error)
- Combine exploration and exploitation in single agents

## Key Concepts

- **Offline RL**: Learning policies from pre-collected datasets without online interaction
- **Conservative Q-Learning (CQL)**: Regularizing Q-values to prevent overestimation in offline settings
- **Advantage-Weighted Actor-Critic (AWAC)**: Weighting policy updates by advantage for stable offline learning
- **Implicit Q-Learning (IQL)**: Learning value functions via expectile regression for offline policy optimization
- **Random Network Distillation (RND)**: Using prediction error as intrinsic reward for exploration
- **Intrinsic vs Extrinsic Rewards**: Balancing curiosity-driven and task-driven learning

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

- Example command:
  ```bash
  python cs285/scripts/run_hw5.py --env_name PointmassMedium-v0 --exp_name awac_test
  ```
- Results are saved in `results/`.

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
