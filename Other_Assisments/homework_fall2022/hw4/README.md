# Homework 4: Model-Based Reinforcement Learning

## Overview

This assignment introduces model-based reinforcement learning, where agents learn a model of the environment's dynamics and use it for planning and policy improvement. You will implement model-based agents using neural network dynamics models, explore model predictive control (MPC), and investigate model-based policy optimization (MBPO) for sample-efficient learning.

## Learning Objectives

- Understand the principles of model-based RL
- Implement neural network dynamics models
- Develop model predictive control algorithms
- Analyze the trade-offs between model-based and model-free approaches
- Explore data augmentation and ensemble methods for robust modeling

## Key Concepts

- **Model-Based RL**: Learning a model of environment dynamics for planning
- **Dynamics Models**: Neural networks that predict next states and rewards
- **Model Predictive Control (MPC)**: Using the model for trajectory optimization
- **Model-Based Policy Optimization (MBPO)**: Combining model-based and model-free learning
- **Uncertainty Estimation**: Using ensembles for better exploration and robustness
- **Sample Efficiency**: Achieving good performance with fewer environment interactions

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

- Example command:
  ```bash
  python cs285/scripts/run_hw4.py --env_name PointmassEasy-v0 --exp_name mb_test
  ```
- Results are saved in `results/`.

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

You may also want to look through [scripts/run_hw4_mb.py](cs285/scripts/run_hw4_mb.py) and [scripts/run_hw4_mbpo.py](cs285/scripts/run_hw4_mbpo.py) (if running locally) or [scripts/run_hw4_mb.ipynb](cs285/scripts/run_hw4_mb.ipynb) and [scripts/run_hw4_mbpo.ipynb](cs285/scripts/run_hw4_mbpo.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

See the [assignment PDF](cs285_hw4.pdf) for more details on what files to edit.
